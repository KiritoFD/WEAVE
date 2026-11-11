"""SD0-S10 V2 acceptance evaluation.

    python -m tools.vae_suite.eval_sd0_s10 --models official,old_s10,c0,c1

Layers 1-2 + infra (layer 3 / generator swap is a separate final step):
  L1 clean-latent reconstruction : L1 / PSNR / SSIM / LPIPS@512 / CSD Keep /
                                   HF correlation / HF energy ratio / colour
  L2 perturbation & Jacobian     : teacher/student image-change cosine, amplitude
                                   ratio, HF change correlation, input-grad cosine
                                   AND norm ratio, for SMALL and MEDIUM bands;
                                   plus J^T v with identical image directions v
  L4 infra                       : forward / forward+input-grad ms + peak GB

All compared on the SAME official SD0 Encoder latents (frozen holdout + test256).
The old 0.1946 A0/S10 LPIPS-gradient number is NOT reused as a baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "vae_suite"))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder
from tools.vae_suite.sd0_losses import haar_dwt, make_perturbation
from tools.vae_suite.train_sd0_s10 import load_data

SD0 = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
WIDTHS = [64, 128, 192, 256]
OUT = ROOT / "runs/sd0_s10"
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]


def build_variant(name, official, dev):
    """Return (callable dec(z), meta). dec applies pqc exactly once."""
    pqc = official.post_quant_conv
    if name == "official":
        return lambda z: official.decoder(pqc(z)), {"params": 49490000}
    ckpath = {
        "old_s10": ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
        "c0": OUT / "c0_best.pt",
        "c1": OUT / "c1_best.pt",
    }.get(name)
    if ckpath is None or not ckpath.exists():
        raise FileNotFoundError(f"{name}: {ckpath}")
    ck = torch.load(ckpath, map_location="cpu", weights_only=False)
    d = build_decoder(WIDTHS)
    d.load_state_dict(ck["state_dict"])
    pq = torch.nn.Conv2d(4, 4, 1)
    pq.load_state_dict(ck["post_quant_conv"])
    sd = StudentDecoder(pq.to(dev), d.to(dev)).to(dev).eval()
    # old_s10 is the HISTORICAL A0-trained S10: it keeps its own A0-trained
    # post_quant_conv on purpose. c0/c1 carry the SD0-trained pqc. This is a
    # deliberate confound in the "old S10" reference, not an oversight -- the
    # plan asks for "SD0 vs old S10 vs new C0 vs new C1" with the SAME SD0
    # latents, so old_s10's A0 pqc is reported here so the comparison is
    # interpretable. All variants receive IDENTICAL unscaled SD0 latents.
    pqc_src = "A0-trained (historical old_s10)" if name == "old_s10" else "SD0-trained"
    return (lambda z: sd(z)), {"params": sum(p.numel() for p in d.parameters()),
                               "ckpt": str(ckpath), "pqc_source": pqc_src,
                               "meta": ck.get("meta", {})}


def csd_keep(csd, y, x, dev):
    nm = torch.tensor(NM, device=dev).view(1, 3, 1, 1)
    ns = torch.tensor(NS, device=dev).view(1, 3, 1, 1)
    f = []
    for t in (y, x):
        t01 = (t.clamp(-1, 1) + 1) / 2
        t224 = F.interpolate(t01, (224, 224), mode="bicubic", align_corners=False)
        _, _, sf = csd((t224 - nm) / ns)
        f.append(F.normalize(sf.float(), dim=-1))
    return float((f[0] * f[1]).sum(-1).mean())


def ssim_simple(x, y):
    a, b = (x.clamp(-1, 1) + 1) / 2, (y.clamp(-1, 1) + 1) / 2
    C1, C2 = 1e-4, 9e-4
    ma, mb = F.avg_pool2d(a, 11, 1, 5), F.avg_pool2d(b, 11, 1, 5)
    maa, mbb, mab = ma * ma, mb * mb, ma * mb
    sa = F.avg_pool2d(a * a, 11, 1, 5) - maa
    sb = F.avg_pool2d(b * b, 11, 1, 5) - mbb
    sab = F.avg_pool2d(a * b, 11, 1, 5) - mab
    return float((((2 * mab + C1) * (2 * sab + C2)) /
                  ((maa + mbb + C1) * (sa + sb + C2))).mean())


@torch.no_grad()
def recon_metrics(dec, ds, dev, lp, csd, n=256, bs=4, tag="holdout"):
    dec_eval = dec
    acc = {k: [] for k in ["l1", "psnr", "ssim", "lpips", "csd",
                            "hf_corr_L0", "hf_energy_ratio", "colour"]}
    idx = np.random.default_rng(31 if tag == "holdout" else 71).choice(
        min(n, len(ds)), min(n, len(ds)), replace=False)
    for i in range(0, len(idx), bs):
        recs = [ds[int(j)] for j in idx[i:i + bs]]
        x = torch.stack([r["x"] for r in recs]).to(dev)
        z = torch.stack([r["z"] for r in recs]).to(dev)
        y = dec_eval(z).float()
        acc["l1"].append(float((y - x).abs().mean()))
        mse = float(F.mse_loss((y.clamp(-1, 1) + 1) / 2, (x + 1) / 2))
        acc["psnr"].append(10 * np.log10(1 / max(mse, 1e-10)))
        acc["ssim"].append(ssim_simple(y, x))
        acc["lpips"].append(float(lp(y.clamp(-1, 1), x).mean()))
        acc["csd"].append(csd_keep(csd, y, x, dev))
        wy, wx = haar_dwt(y, 1)[0], haar_dwt(x, 1)[0]
        cs, er = [], []
        for bi in range(3):
            a, c = wy[bi].flatten(), wx[bi].flatten()
            a2, c2 = a - a.mean(), c - c.mean()
            cs.append(float((a2 * c2).sum() / (a2.norm() * c2.norm() + 1e-8)))
            er.append(float(wy[bi].std() / (wx[bi].std() + 1e-8)))
        acc["hf_corr_L0"].append(float(np.mean(cs)))
        acc["hf_energy_ratio"].append(float(np.mean(er)))
        acc["colour"].append(float((y.mean((2, 3)) - x.mean((2, 3))).abs().mean()))
        del x, z, y
        torch.cuda.empty_cache()
    return {k: round(float(np.mean(v)), 5) for k, v in acc.items()}


def perturb_metrics(student, teacher, ds, dev, lp, n=64, bs=4, seed=0):
    """L2: small (0.02,0.05) vs medium (0.10,0.20) perturbation bands."""
    out = {}
    gen = torch.Generator(device=dev).manual_seed(seed)
    idx = np.random.default_rng(seed).choice(min(n, len(ds)), min(n, len(ds)),
                                             replace=False)
    for band, fracs in (("small", [0.02, 0.05]), ("medium", [0.10, 0.20])):
        rows = []
        for i in range(0, len(idx), bs):
            recs = [ds[int(j)] for j in idx[i:i + bs]]
            x = torch.stack([r["x"] for r in recs]).to(dev)
            z = torch.stack([r["z"] for r in recs]).to(dev)
            for frac in fracs:
                for kind in ("corr_gauss", "interp"):
                    partner = torch.roll(z, 1, dims=0) if kind == "interp" else None
                    delta = make_perturbation(z, kind, frac, partner, generator=gen)
                    # Evaluation only: BOTH networks under no_grad, otherwise
                    # student's autograd graphs accumulate across the loop and
                    # leak GPU memory. The input-gradient section below builds
                    # its own fresh graphs explicitly.
                    with torch.no_grad():
                        dy_t = teacher(z + delta) - teacher(z)
                        dy_s = student(z + delta) - student(z)
                    cs = float(F.cosine_similarity(dy_s.flatten(1),
                                                   dy_t.flatten(1), dim=1).mean())
                    amp = float(dy_s.abs().mean() / (dy_t.abs().mean() + 1e-8))
                    # HF change correlation on finest band
                    wy, wt = haar_dwt(dy_s, 1)[0], haar_dwt(dy_t, 1)[0]
                    hc = []
                    for bi in range(3):
                        a, b = wy[bi].flatten(), wt[bi].flatten()
                        a, b = a - a.mean(), b - b.mean()
                        hc.append(float((a * b).sum() / (a.norm() * b.norm() + 1e-8)))
                    rows.append({"frac": frac, "kind": kind, "cos": cs,
                                 "amp_ratio": amp, "hf_corr": float(np.mean(hc))})
            del x, z
            torch.cuda.empty_cache()
        out[band] = {
            "cos": round(float(np.mean([r["cos"] for r in rows])), 4),
            "amp_ratio": round(float(np.mean([r["amp_ratio"] for r in rows])), 4),
            "hf_corr": round(float(np.mean([r["hf_corr"] for r in rows])), 4),
            "n": len(rows),
        }

    # ---- input-gradient (two distinct types) ----
    # The official 49.49M teacher needs a 512px autograd graph for LPIPS and
    # J^T v. At 8 images this OOMs at 21.57 GB (observed in the P5 run), so use
    # only 2 images and free every intermediate graph immediately. n=2 is enough
    # for a per-image cosine / norm ratio; the metric is averaged over that set.
    NG = 2
    recs = [ds[int(j)] for j in idx[:NG]]
    x = torch.stack([r["x"] for r in recs]).to(dev)
    z0 = torch.stack([r["z"] for r in recs]).to(dev)
    # (a) full training-Loss input gradient: LPIPS(D(z), x)
    grads_loss = {}
    for nm, dec in (("teacher", teacher), ("student", student)):
        zz = z0.clone().requires_grad_(True)
        l = lp(dec(zz).clamp(-1, 1), x).mean()
        g, = torch.autograd.grad(l, zz)
        grads_loss[nm] = g.detach().flatten(1).float()
        del zz, l, g
        torch.cuda.empty_cache()
    # (b) Jacobian-transpose response J^T v with IDENTICAL image directions v
    jt = {"teacher": [], "student": []}
    for di in range(4):
        torch.manual_seed(1000 + di)
        # identical multi-scale smooth image direction v for BOTH networks
        v = F.interpolate(torch.randn(x.shape[0], 3, 32, 32, device=dev,
                                      generator=torch.Generator(device=dev)
                                      .manual_seed(2000 + di)),
                          size=(512, 512), mode="bilinear", align_corners=False)
        for nm, dec in (("teacher", teacher), ("student", student)):
            zz = z0.clone().requires_grad_(True)
            y = dec(zz)
            g, = torch.autograd.grad((y * v).sum(), zz)
            jt[nm].append(g.detach().flatten(1).float())
            del zz, y, g
            torch.cuda.empty_cache()
        del v
    jt_cos = [float(F.cosine_similarity(a, b, dim=1).mean())
              for a, b in zip(jt["teacher"], jt["student"])]
    out["input_grad"] = {
        "loss_grad_cos": round(float(F.cosine_similarity(
            grads_loss["teacher"], grads_loss["student"], dim=1).mean()), 4),
        "loss_grad_norm_ratio": round(float(
            grads_loss["student"].norm(dim=1).mean() /
            (grads_loss["teacher"].norm(dim=1).mean() + 1e-8)), 4),
        "jtv_cos_mean": round(float(np.mean(jt_cos)), 4),
        "jtv_cos_median": round(float(np.median(jt_cos)), 4),
        "jtv_cos_all": [round(c, 4) for c in jt_cos],
    }
    return out


def infra(dec, dev, iters=10):
    """Forward and forward+input-gradient timing.

    NOTE: cannot be wrapped in @torch.no_grad() -- the "forward+grad" mode
    must build an autograd graph to call .backward(). Only the plain "forward"
    mode is run under no_grad.
    """
    r = {}
    for mode in ("forward", "forward+grad"):
        z = torch.randn(1, 4, 64, 64, device=dev)
        grad_mode = (mode == "forward+grad")
        if grad_mode:
            z.requires_grad_(True)
        ctx = torch.enable_grad() if grad_mode else torch.no_grad()
        with ctx:
            for _ in range(3):
                y = dec(z)
                if grad_mode:
                    y.float().abs().mean().backward()
            if dev.startswith("cuda"):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(iters):
                if grad_mode:
                    z.grad = None
                y = dec(z)
                if grad_mode:
                    y.float().abs().mean().backward()
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() / 2**30
                    if dev.startswith("cuda") else 0.0)
            r[mode] = {"ms": round((time.time() - t0) / iters * 1000, 2),
                       "peak_GB": round(peak, 2)}
        del z
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="official,old_s10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--skip-response", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = args.device
    sp, ds_tr, ds_ho, ds_te = load_data()
    official = AutoencoderKL.from_pretrained(SD0, torch_dtype=torch.float32).to(dev).eval()
    for p in official.parameters():
        p.requires_grad_(False)
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    import csd_model
    csd, _, _ = csd_model.load_csd(dev); csd = csd.float().eval()

    teacher = lambda z: official.decoder(official.post_quant_conv(z))
    results = {}
    for name in args.models.split(","):
        name = name.strip()
        if name == "teacher":
            continue
        print(f"\n=== {name} ===", flush=True)
        dec, meta = build_variant(name, official, dev)
        r = {"meta": meta,
             "recon_holdout": recon_metrics(dec, ds_ho, dev, lp, csd,
                                             n=args.n, tag="holdout"),
             "recon_test256": recon_metrics(dec, ds_te, dev, lp, csd,
                                             n=min(256, args.n), tag="test256"),
             "infra": infra(dec, dev)}
        print(f"  holdout {json.dumps(r['recon_holdout'])}", flush=True)
        print(f"  test256 {json.dumps(r['recon_test256'])}", flush=True)
        if not args.skip_response and name != "official":
            r["perturb"] = perturb_metrics(dec, teacher, ds_ho, dev, lp,
                                            n=64, seed=0)
            print(f"  perturb {json.dumps(r['perturb'])}", flush=True)
        results[name] = r
        del dec
        torch.cuda.empty_cache()

    # paired deltas vs official (clean-latent reconstruction)
    if "official" in results:
        base = results["official"]["recon_holdout"]
        for name, r in results.items():
            if name == "official":
                continue
            r["delta_vs_official"] = {
                k: round(r["recon_holdout"][k] - base[k], 5)
                for k in base if k in r["recon_holdout"]}

    print("\n" + "=" * 100)
    print("ACCEPTANCE SUMMARY (holdout)")
    print("=" * 100)
    print(f"{'model':<12}{'L1':>9}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>9}"
          f"{'CSD':>8}{'HFcorr':>8}{'HFenergy':>9}")
    for name, r in results.items():
        v = r["recon_holdout"]
        print(f"{name:<12}{v['l1']:>9.5f}{v['psnr']:>8.2f}{v['ssim']:>8.4f}"
              f"{v['lpips']:>9.5f}{v['csd']:>8.4f}{v['hf_corr_L0']:>8.4f}"
              f"{v['hf_energy_ratio']:>9.4f}")
    tag = args.tag or "acceptance"
    outp = OUT / f"eval_{tag}.json"
    outp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"\n[Saved] {outp}", flush=True)


if __name__ == "__main__":
    main()
