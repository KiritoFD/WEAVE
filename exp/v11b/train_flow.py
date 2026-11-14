"""Train the style-conditioned wavelet flow (V11-B P2).

Flow matching on real-painting wavelet detail coefficients:
  h_tau = (1-tau)*eps + tau*h_real,  v* = h_real - eps
Loss = MSE(v_theta(h_tau, LL_match, edge_match, style, tau), v*)
Classifier-free style dropout p=0.15. Both bands (H1,H2) per step.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .bands import dwt2, sobel_mag
from .dataset_real import (RamWaveletDataset, RealWaveletDataset,
                           collate_batch, make_batch)
from .model import WaveletFlow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "v11b"


def estimate_sigmas(n_images: int = 400, device="cuda") -> dict:
    """Per-style per-band coefficient RMS on train crops (fp32)."""
    ds = RealWaveletDataset("train", crop=128)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    sums = {s: {"H1": 0., "H2": 0., "n": 0} for s in range(5)}
    loader_items = [ds[i] for i in idx]
    for c0 in range(0, len(loader_items), 16):
        batch = loader_items[c0:c0 + 16]
        crops = torch.stack([b[0] for b in batch])
        sids = torch.tensor([b[1] for b in batch])
        b = make_batch(crops, sids, device)
        for sid in range(5):
            m = sids == sid
            if not m.any():
                continue
            for band in ("H1", "H2"):
                v = b[band][m].float()
                sums[sid][band] += v.pow(2).mean().item() * m.sum().item()
                sums[sid]["n"] += m.sum().item()
    out = {}
    for s in range(5):
        n = max(sums[s]["n"], 1)
        out[str(s)] = {"H1": math.sqrt(sums[s]["H1"] / n),
                       "H2": math.sqrt(sums[s]["H2"] / n)}
    return out


def fm_step_full(model, batch, sigmas, device, p_drop=0.15):
    """Proper version: conditional pass for all, unconditional pass only
    for dropped rows; loss always against v*."""
    total = 0.0
    parts = {}
    for which, band, ll_key, edge_key in (
            ("H1", "H1", "LL1", "edge1"), ("H2", "H2", "LL2", "edge2")):
        h = batch[band].float()
        B = h.shape[0]
        sid = batch["style_id"]
        sig = torch.tensor(
            [sigmas[str(int(s))][band] for s in sid],
            device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        eps = torch.randn_like(h) * sig
        tau = torch.rand(B, device=device)
        ta = tau.view(-1, 1, 1, 1)
        h_tau = (1 - ta) * eps + ta * h
        v_star = h - eps
        drop = torch.rand(B, device=device) < p_drop
        # conditional velocity (all rows; dropped rows will be replaced)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(device) == "cuda"):
            v_cond = model.forward_band(
                which, h_tau, batch[ll_key], batch[edge_key], sid, tau,
                unconditional=False).float()
        if drop.any():
            idx = drop.nonzero(as_tuple=True)[0]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=str(device) == "cuda"):
                v_unc = model.forward_band(
                    which, h_tau[idx], batch[ll_key][idx],
                    batch[edge_key][idx], sid[idx], tau[idx],
                    unconditional=True).float()
            v_cond[idx] = v_unc
        loss = F.mse_loss(v_cond, v_star)
        parts[band] = loss.item()
        total = total + loss
    return total / 2, parts


@torch.no_grad()
def g1_eval(model, sigmas, device, n=128):
    """Holdout velocity-MSE: correct vs wrong vs unconditional style."""
    ds = RealWaveletDataset("holdout", crop=128)
    g = torch.Generator().manual_seed(123)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    items = [ds[i] for i in idx]
    res = {}
    model.eval()
    for which, band, ll_key, edge_key in (
            ("H1", "H1", "LL1", "edge1"), ("H2", "H2", "LL2", "edge2")):
        errs = {"correct": [], "wrong": [], "uncond": []}
        for c0 in range(0, len(items), 16):
            batch = items[c0:c0 + 16]
            crops = torch.stack([b[0] for b in batch])
            sids = torch.tensor([b[1] for b in batch]).to(device)
            bb = make_batch(crops, sids, device)
            h = bb[band].float()
            B = h.shape[0]
            sig = torch.tensor(
                [sigmas[str(int(s))][band] for s in sids],
                device=device).view(-1, 1, 1, 1)
            eps = torch.randn_like(h) * sig
            tau = torch.rand(B, device=device)
            ta = tau.view(-1, 1, 1, 1)
            h_tau = (1 - ta) * eps + ta * h
            v_star = h - eps
            wrong = (sids + 1 + torch.randint(0, 4, (B,), device=device)) % 5
            for name, sid_in, unc in (
                    ("correct", sids, False), ("wrong", wrong, False),
                    ("uncond", sids, True)):
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(device) == "cuda"):
                    v = model.forward_band(
                        which, h_tau, bb[ll_key], bb[edge_key], sid_in,
                        tau, unconditional=unc).float()
                errs[name].append(F.mse_loss(v, v_star, reduction="none")
                                  .mean(dim=(1, 2, 3)))
        res[band] = {k: torch.cat(v).mean().item() for k, v in errs.items()}
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)   # eff batch 32
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--p-drop", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=OUT / "flow_v1")
    ap.add_argument("--sigma", type=Path, default=None,
                    help="json of per-style band sigmas; estimated if absent")
    ap.add_argument("--g1-every", type=int, default=500)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    if args.sigma and args.sigma.is_file():
        sigmas = json.loads(args.sigma.read_text())
        if "sigmas" in sigmas:  # P0_PROFILE.json wrapper
            sigmas = sigmas["sigmas"]
    else:
        print("[p2] estimating per-style band sigmas (400 imgs)...",
              flush=True)
        sigmas = estimate_sigmas(device=device)
        (args.out / "sigmas.json").write_text(json.dumps(sigmas, indent=1))
        print("sigmas:", json.dumps(sigmas)[:400], flush=True)

    model = WaveletFlow(width=args.width).to(device)
    print(f"[p2] params={model.n_params()/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.updates,
        pct_start=0.05)

    ds = RamWaveletDataset("train", crop=args.crop)
    print(f"[p2] train paintings={len(ds)} (RAM-resident)", flush=True)
    loader = DataLoader(ds, batch_size=args.micro, shuffle=True,
                        num_workers=0, collate_fn=collate_batch,
                        drop_last=True, pin_memory=True,
                        persistent_workers=False)

    curve = args.out / "train_curves.jsonl"
    state_path = args.out / "state.json"
    step = 0
    t0 = time.time()
    done = False
    while not done:
        for crops, sids in loader:
            crops, sids = crops.to(device), sids.to(device)
            bb = make_batch(crops, sids, device)
            loss, parts = fm_step_full(model, bb, sigmas, device,
                                       p_drop=args.p_drop)
            (loss / args.accum).backward()
            step += 1
            if step % args.accum == 0 or step == args.updates:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                rec = {"step": step, "loss": loss.item(),
                       "H1": parts.get("H1"), "H2": parts.get("H2"),
                       "sec": round(time.time() - t0, 1)}
                with curve.open("a") as fp:
                    fp.write(json.dumps(rec) + "\n")
                if step % 100 == 0:
                    print(f"[p2] step {step}/{args.updates} "
                          f"loss={loss.item():.4f} H1={parts.get('H1'):.4f} "
                          f"H2={parts.get('H2'):.4f} "
                          f"{(time.time()-t0)/step:.2f}s/step", flush=True)
                if step % args.g1_every == 0:
                    g1 = g1_eval(model, sigmas, device)
                    print(f"[g1@{step}] {json.dumps(g1)}", flush=True)
                    with (args.out / "g1.jsonl").open("a") as fp:
                        fp.write(json.dumps({"step": step, **g1}) + "\n")
                frac = step / args.updates
                for mark, tag in ((0.25, "25"), (0.5, "50"),
                                  (0.75, "75"), (1.0, "100")):
                    if frac >= mark and not (args.out / f"ckpt_{tag}.pt"
                                             ).exists():
                        torch.save({"model": model.state_dict(),
                                    "args": vars(args),
                                    "sigmas": sigmas, "step": step},
                                   args.out / f"ckpt_{tag}.pt")
                        print(f"[ckpt] {tag}% saved", flush=True)
                state_path.write_text(json.dumps(
                    {"step": step, "updates": args.updates,
                     "elapsed_s": time.time() - t0}))
            if step >= args.updates:
                done = True
                break
    print(f"[p2] done step={step} elapsed={time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
