"""P5: H2 Masked Style-Bridge — retrain ONLY the H2 branch as a local editor.

Task per real RGB crop (style s):
  1. h = H2 band of crop; stats per style precomputed from train crops.
  2. h_tilde = appearance-perturbed H2 toward foreign style r (mean/std
     alignment, strength {0,.5,1}) + random block masking (detail dropout).
  3. h_tau0 = (1-tau0)*eps + tau0*h_tilde ; tau0 in {0.4,0.7}
  4. Bridge FM to real h with v* = (h - h_tau0)/(1-tau0), condition = s.

H1 branch frozen (not trained, not in loss). Init from --source ckpt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

from exp.v11b.bands import dwt2, sobel_mag
from exp.v11b.model import WaveletFlow
from exp.v15.h2_train_4070lp import collect_images, STYLES

REPO = Path(__file__).resolve().parents[2]


def band_stats(X, y, n_per_style=200, device="cuda"):
    """Per-style H2 mean/std over random crops (already-cropped X on CPU)."""
    mu = torch.zeros(5, 12); sd = torch.ones(5, 12)
    g = torch.Generator().manual_seed(0)
    for s in range(5):
        idx = (y == s).nonzero(as_tuple=True)[0]
        take = idx[torch.randperm(len(idx), generator=g)[:n_per_style]]
        crops = X[take].to(device)
        d = dwt2(crops)
        h = d["H2"].flatten(1)                      # (n,12,H*W)-> use mean over HW
        h3 = d["H2"]
        mu[s] = h3.mean(dim=(0, 2, 3))
        sd[s] = h3.std(dim=(0, 2, 3)).clamp_min(1e-4)
        del crops, d, h3
    return mu.to(device), sd.to(device)


def perturb_h2(h, sid, rid, strength, mu, sd, mask_frac=0.25,
               device="cuda", gen=None):
    """h: (B,12,h,w) H2 band; align channel stats toward style rid, blend
    by strength, then randomly replace blocks with noise."""
    B = h.shape[0]
    m = mu[sid]                                     # (B,12)
    s_ = sd[sid]
    mr = mu[rid]
    sr = sd[rid]
    hn = (h - m.view(B, 12, 1, 1)) / s_.view(B, 12, 1, 1)
    hp = hn * sr.view(B, 12, 1, 1) + mr.view(B, 12, 1, 1)
    out = h + strength.view(B, 1, 1, 1) * (hp - h)
    # block mask: replace masked 8x8 cells with gaussian noise at band scale
    H, W = out.shape[-2:]
    bh, bw = max(H // 8, 1), max(W // 8, 1)
    grid = torch.rand(B, 1, bh, bw, device=device, generator=gen) < mask_frac
    grid = Fn.interpolate(grid.float(), size=(H, W), mode="nearest") > 0.5
    noise = torch.randn_like(out) * s_.view(B, 12, 1, 1)
    return torch.where(grid, noise, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--image-roots", nargs="+", required=True)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--width", type=int, default=56)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mask-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-style", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sigmas = json.loads((REPO / "docs/v11b/P0_PROFILE.json")
                        .read_text())["sigmas"]
    ck = torch.load(a.source, map_location="cpu", weights_only=False)
    model = WaveletFlow(width=a.width).to(dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    # freeze everything except H2 branch
    for n, p in model.named_parameters():
        p.requires_grad_("net_h2" in n)
    h2_params = [p for n, p in model.named_parameters()
                 if "net_h2" in n]
    print(f"trainable H2 params: {sum(p.numel() for p in h2_params)}",
          flush=True)
    opt = torch.optim.AdamW(h2_params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.updates)

    print("loading images...", flush=True)
    X, y = collect_images(a.image_roots, a.crop, dev,
                          a.max_per_style or None)
    print(f"images {X.shape} styles={torch.bincount(y).tolist()}",
          flush=True)
    mu, sd = band_stats(X, y, device=dev)

    curve = out / "train_curves.jsonl"
    opt_step = micro_in = 0
    run = 0.0
    t0 = time.time()
    peak = 0
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    while opt_step < a.updates:
        idx = torch.randint(0, len(X), (a.micro,))
        crops = X[idx].to(dev)
        d = dwt2(crops)
        h_real = d["H2"].float()
        ll2, edge2 = d["LL2"], sobel_mag(d["LL2"].mean(1, keepdim=True))
        sid = y[idx].to(dev)
        rid = (sid + 1 + torch.randint(0, 4, (len(idx),), device=dev,
                                       generator=gen)) % 5
        strength = torch.tensor([0.0, 0.5, 1.0], device=dev)[
            torch.randint(0, 3, (len(idx),), device=dev, generator=gen)]
        tau0 = torch.tensor([0.4, 0.7], device=dev)[
            torch.randint(0, 2, (len(idx),), device=dev, generator=gen)]
        h_tilde = perturb_h2(h_real, sid, rid, strength, mu, sd,
                             a.mask_frac, dev, gen)
        t0v = tau0.view(-1, 1, 1, 1)
        eps = torch.randn_like(h_real) * \
            torch.tensor([sigmas[str(int(s))]["H2"] for s in sid],
                         device=dev).view(-1, 1, 1, 1)
        h_tau0 = (1 - t0v) * eps + t0v * h_tilde
        # per-row tau in [tau0,1]
        tau = tau0 + (1 - tau0) * torch.rand(len(idx), device=dev,
                                             generator=gen)
        span = (1 - tau0).clamp_min(1e-6)
        aa = ((1 - tau) / span).view(-1, 1, 1, 1)
        bb = ((tau - tau0) / span).view(-1, 1, 1, 1)
        h_t = aa * h_tau0 + bb * h_real
        v_star = (h_real - h_tau0) / span.view(-1, 1, 1, 1)
        drop = torch.rand(len(idx), device=dev) < 0.15
        with torch.autocast(dev.type, dtype=torch.bfloat16,
                            enabled=dev.type == "cuda"):
            v = model.forward_band("H2", h_t, ll2, edge2, sid, tau,
                                   unconditional=False).float()
        if drop.any():
            idd = drop.nonzero(as_tuple=True)[0]
            with torch.autocast(dev.type, dtype=torch.bfloat16,
                                enabled=dev.type == "cuda"):
                v[idd] = model.forward_band(
                    "H2", h_t[idd], ll2[idd], edge2[idd], sid[idd],
                    tau[idd], unconditional=True).float()
        loss = Fn.mse_loss(v, v_star) / a.accum
        loss.backward()
        run += loss.item() * a.accum
        micro_in += 1
        if micro_in % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(h2_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            peak = max(peak, torch.cuda.memory_allocated() // 2**20)
            if opt_step % 100 == 0:
                rec = {"opt_step": opt_step,
                       "loss": round(run / (a.accum * 100), 5),
                       "sec": round(time.time() - t0, 1),
                       "peak_mb": peak}
                with curve.open("a") as fp:
                    fp.write(json.dumps(rec) + "\n")
                print(f"[h2bridge] {opt_step}/{a.updates} "
                      f"loss={rec['loss']} {rec['sec']}s peak={peak}MB",
                      flush=True)
                run = 0.0
    torch.save({"model": model.state_dict(), "args": vars(a),
                "sigmas": sigmas, "step": opt_step}, out / "final.pt")
    print(f"DONE {opt_step} {time.time()-t0:.0f}s peak={peak}MB", flush=True)


if __name__ == "__main__":
    main()
