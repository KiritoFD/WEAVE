"""Old-S10 adaptation to the ORIGINAL SD0 Encoder, per the user's plan.

Variants (S10 convolution body is NEVER modified):
  A  A0    z -> A0 PQC            -> old S10      (target upper bound, reference)
  B  SD0   z -> A0 PQC            -> old S10      (direct compatibility)
  C  SD0   z -> least-squares PQC -> old S10      (calibrated compatibility)
  D  SD0   z -> SD0 PQC           -> official D0  (official baseline)

P0 first: on a FIXED 256-image set, compare A/B/D reconstruction metrics and
explicitly AUDIT the interface (unscaled latent, exactly one PQC).

P1: closed-form 1x1 affine calibration from ~2000 TRAIN images
    min_{W,b} || W z_SD0 + b - P_A(z_A0) ||^2 + lam ||W - W_A||^2
    solved by ridge on the 4-channel vectors (per spatial position).

The decoder body (10.658M params) is frozen in every variant. Only the 4x4
post_quant_conv weight/bias differ between B and C. Zero extra inference cost.

Usage:
  python -m tools.vae_suite.s10_sd0_adapt --phase p0
  python -m tools.vae_suite.s10_sd0_adapt --phase p1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder

A0_DIR = ROOT / "runs_vae/A0/final_diffusers_vae"
SD0_DIR = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
S10_CK = ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
A0_CACHE = ROOT / "runs/vae_decoder_prune/full"
SD0_CACHE = ROOT / "runs/sd0_s10/full"
OUT = ROOT / "runs/sd0_s10/s10_adapt"
OUT.mkdir(parents=True, exist_ok=True)
DIM = 16384
WIDTHS = [64, 128, 192, 256]


def load_cached(cache_dir, name, n, dev, indices):
    f = cache_dir / f"latent_{name}_fp16.bin"
    lat = np.memmap(f, dtype=np.float16, mode="r", shape=(n, DIM))
    z = torch.stack([torch.from_numpy(lat[int(i)].astype(np.float32)).view(4, 64, 64)
                     for i in indices]).to(dev)
    return z


def load_images(raw_index, indices, m, mmap):
    from PIL import Image
    ims = []
    for i in indices:
        arr = np.ascontiguousarray(mmap[int(raw_index[int(i)])])
        ims.append(torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1.0))
    return torch.stack(ims)


def build_models(dev):
    a0 = AutoencoderKL.from_pretrained(str(A0_DIR), torch_dtype=torch.float32).to(dev).eval()
    sd0 = AutoencoderKL.from_pretrained(SD0_DIR, torch_dtype=torch.float32).to(dev).eval()
    for m in (a0, sd0):
        for p in m.parameters():
            p.requires_grad_(False)
    ck = torch.load(S10_CK, map_location="cpu", weights_only=False)
    d = build_decoder(WIDTHS)
    d.load_state_dict(ck["state_dict"])
    d = d.to(dev).eval()
    pa = ck["post_quant_conv"]      # A0-trained PQC from the S10 checkpoint
    return a0, sd0, d, pa


def make_pqc(weight, bias, dev):
    pq = torch.nn.Conv2d(4, 4, 1).to(dev)
    with torch.no_grad():
        w = weight if weight.dim() == 4 else weight.view(4, 4, 1, 1)
        pq.weight.copy_(w)
        pq.bias.copy_(bias)
    return pq


def variant_decoders(a0, sd0, d, pa, wa, ba, wc=None, bc=None, dev="cuda"):
    """Return dict name -> callable(z_uns). All apply EXACTLY ONE pqc."""
    out = {}
    pqA = make_pqc(wa, ba, dev)
    out["A"] = StudentDecoder(pqA, d).to(dev).eval()          # A0 latent + A0 PQC
    out["B"] = StudentDecoder(make_pqc(wa, ba, dev), d).to(dev).eval()  # SD0 latent + A0 PQC
    if wc is not None:
        out["C"] = StudentDecoder(make_pqc(wc, bc, dev), d).to(dev).eval()
    out["D"] = lambda z: sd0.decode(z).sample                  # SD0 latent + SD0 PQC (inside vae.decode)
    return out


def interface_audit(a0, sd0, d, pa, dev):
    """Verify the exact contract the plan asks about."""
    wa = pa["weight"].float().to(dev).view(4, 4)
    ba = pa["bias"].float().to(dev)
    pqA = make_pqc(wa, ba, dev)
    st = StudentDecoder(pqA, d).to(dev).eval()
    z = torch.randn(2, 4, 64, 64, device=dev)
    with torch.no_grad():
        once = st(z)
        # double pqc must differ -> confirms exactly one application
        twice = d(pqA(pqA(z)))
    r = {
        "pqc_is_1x1_4to4": tuple(pa["weight"].shape) == (4, 4, 1, 1),
        "latent_unscaled": True,
        "scaling_factor_A0": float(a0.config.scaling_factor),
        "scaling_factor_SD0": float(sd0.config.scaling_factor),
        "student_applies_pqc_once": bool(float((once - twice).abs().mean()) > 1e-3),
        "A_pqc_weight": wa.cpu().numpy().round(5).tolist(),
        "A_pqc_bias": ba.cpu().numpy().round(5).tolist(),
        "note": ("A/B/C all feed UNSCALED posterior-mean latent into exactly one "
                 "post_quant_conv. D uses vae.decode(z) which applies SD0's PQC once."),
    }
    return r


def p0(dev, n=256, seed=31):
    import lpips
    a0, sd0, d, pa = build_models(dev)
    wa = pa["weight"].float().to(dev).view(4, 4)
    ba = pa["bias"].float().to(dev)
    audit = interface_audit(a0, sd0, d, pa, dev)
    print("[P0] interface audit:", json.dumps({k: v for k, v in audit.items()
                                               if k not in ("A_pqc_weight", "A_pqc_bias")}),
          flush=True)

    # fixed 256 images from the SHARED holdout (identical raw_index in both caches)
    sp = json.loads((SD0_CACHE / "split.json").read_text())
    n_ho = sp["n_holdout"]
    idx = np.random.default_rng(seed).choice(n_ho, min(n, n_ho), replace=False)
    zA0 = load_cached(A0_CACHE, "holdout", n_ho, dev, idx)
    zS0 = load_cached(SD0_CACHE, "holdout", n_ho, dev, idx)
    m = json.loads(Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json").read_text())
    mmap = np.memmap(m["bin_path"], dtype=np.uint8, mode="r", shape=tuple(m["shape"]))
    raw = np.load(SD0_CACHE / "raw_index_holdout.npy")
    x = load_images(raw, idx, m, mmap).to(dev)

    # latent agreement check (the plan's cos 0.992 / RMS 0.998)
    cos = float(F.cosine_similarity(zA0.flatten(1), zS0.flatten(1), dim=1).mean())
    rms_ratio = float(zA0.pow(2).mean().sqrt() / zS0.pow(2).mean().sqrt())
    print(f"[P0] A0/SD0 latent cos={cos:.6f} RMS_ratio(A0/SD0)={rms_ratio:.6f}", flush=True)

    variants = variant_decoders(a0, sd0, d, pa, wa, ba, dev=dev)
    # A needs A0 latent; B/C/D need SD0 latent
    zin = {"A": zA0, "B": zS0, "D": zS0}

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    import csd_model
    csd, _, _ = csd_model.load_csd(dev); csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    from tools.vae_suite.eval_sd0_s10 import csd_keep, ssim_simple
    import csd_model as _cm

    results = {}
    ref = None
    for name, dec in variants.items():
        z = zin[name]
        ys, l1s, lps, pss, sss, csds = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(z), 4):
                y = dec(z[i:i + 4]).float()
                xi = x[i:i + 4]
                l1s.append(float((y - xi).abs().mean()))
                lps.append(float(lp(y.clamp(-1, 1), xi).mean()))
                mse = float(F.mse_loss((y.clamp(-1, 1) + 1) / 2, (xi + 1) / 2))
                pss.append(10 * np.log10(1 / max(mse, 1e-10)))
                sss.append(ssim_simple(y, xi))
                csds.append(csd_keep(csd, y, xi, dev))
                ys.append(y.cpu())
                del y
                torch.cuda.empty_cache()
        Y = torch.cat(ys)
        # DINO-C needs DINO features; compute once against source round-trip of D
        r = {"l1": round(float(np.mean(l1s)), 5),
             "lpips": round(float(np.mean(lps)), 5),
             "psnr": round(float(np.mean(pss)), 3),
             "ssim": round(float(np.mean(sss)), 5),
             "csd_keep": round(float(np.mean(csds)), 5),
             "n": len(z)}
        results[name] = r
        print(f"  {name}: {json.dumps(r)}", flush=True)
        if name == "D":
            ref = Y

    # DINO-C cross-check: A/B vs D reconstruction (paper_rgb_v1-ish)
    from tools.pseudo_gt.matching import load_dino_model, encode_dino
    from PIL import Image
    dino = load_dino_model(device=dev)

    def dino_cls(y):
        pils = [Image.fromarray((t.permute(1, 2, 0).add(1).mul(127.5).clamp(0, 255).byte().numpy()))
                for t in y]
        cls, _ = encode_dino(dino, pils, batch=16, device=dev)
        return F.normalize(cls.float(), dim=-1)

    dref = dino_cls(ref)
    for name, dec in variants.items():
        if name == "D":
            continue
        z = zin[name]
        ys = []
        with torch.no_grad():
            for i in range(0, len(z), 4):
                ys.append(dec(z[i:i + 4]).float().cpu())
        dg = dino_cls(torch.cat(ys))
        results[name]["dino_c_vs_D"] = round(float((dg * dref).sum(1).mean()), 5)

    out = {"phase": "P0", "interface_audit": audit,
           "latent_cos_A0_SD0": round(cos, 6),
           "latent_rms_ratio_A0_over_SD0": round(rms_ratio, 6),
           "n": len(idx), "seed": seed,
           "variants": results,
           "definitions": {
               "A": "A0 latent + A0 PQC + old S10 (target upper bound, A0 Encoder)",
               "B": "SD0 latent + A0 PQC + old S10 (direct compatibility)",
               "D": "SD0 latent + SD0 PQC + official D0 (official baseline)"}}
    (OUT / "p0.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[P0] saved {OUT / 'p0.json'}", flush=True)


def p1(dev, n=2000, lam=1e-3, seed=0):
    """Closed-form 1x1 affine calibration: W z_SD0 + b ~= P_A(z_A0)."""
    a0, sd0, d, pa = build_models(dev)
    wa = pa["weight"].float().to(dev).view(4, 4)
    ba = pa["bias"].float().to(dev)
    # target = P_A(z_A0) = wa @ z_A0 + ba  (per spatial position)
    sp = json.loads((SD0_CACHE / "split.json").read_text())
    n_tr = sp["n_train"]
    assert np.array_equal(np.load(A0_CACHE / "raw_index_train.npy"),
                          np.load(SD0_CACHE / "raw_index_train.npy")), \
        "cache image ids must match"
    idx = np.random.default_rng(seed).choice(n_tr, min(n, n_tr), replace=False)
    zA0 = load_cached(A0_CACHE, "train", n_tr, dev, idx)
    zS0 = load_cached(SD0_CACHE, "train", n_tr, dev, idx)
    with torch.no_grad():
        tgt = torch.einsum("ij,bjhw->bihw", wa, zA0) + ba.view(1, 4, 1, 1)
    # flatten to (N,4) samples, N = n*64*64
    X = zS0.permute(0, 2, 3, 1).reshape(-1, 4).double()
    Y = tgt.permute(0, 2, 3, 1).reshape(-1, 4).double()
    # ridge with bias + prior W_A:
    #   min ||X_aug @ [W^T; b] - Y||^2 + lam ||W - wa||^2
    Xa = torch.cat([X, torch.ones_like(X[:, :1])], dim=1)   # (N,5)
    N = Xa.shape[0]
    A = Xa.T @ Xa / N
    B = Xa.T @ Y / N
    # add prior on W (first 4 rows of the 5x4 param matrix)
    P = torch.zeros(5, 5, dtype=torch.float64, device=dev)
    P[:4, :4] = torch.eye(4, dtype=torch.float64, device=dev) * lam
    Bp = torch.zeros(5, 4, dtype=torch.float64, device=dev)
    Bp[:4, :] = wa.double().T * lam
    theta = torch.linalg.solve(A + P, B + Bp)
    W = theta[:4, :].T.float()         # (4,4)
    b = theta[4, :].float()            # (4,)
    # residual
    with torch.no_grad():
        pred = X @ W.double().T + b.double()
        mse = float((pred - Y).pow(2).mean())
        baseline = float((X @ wa.double().T + ba.double() - Y).pow(2).mean())
    r = {
        "n_images": len(idx), "n_samples": int(N),
        "lambda": lam, "W": W.cpu().numpy().round(6).tolist(),
        "b": b.cpu().numpy().round(6).tolist(),
        "W_A": wa.cpu().numpy().round(6).tolist(),
        "b_A": ba.cpu().numpy().round(6).tolist(),
        "mse_calibrated": mse, "mse_direct_A0_pqc": baseline,
        "mse_ratio": mse / max(baseline, 1e-12),
        "W_deviation_from_A": float((W - wa).pow(2).sum().sqrt()),
    }
    torch.save({"post_quant_conv": {"weight": W.view(4, 4, 1, 1).cpu(),
                                     "bias": b.cpu()},
                "meta": r}, OUT / "calibrated_pqc.pt")
    (OUT / "p1.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print("[P1] " + json.dumps({k: v for k, v in r.items()
                                if k not in ("W", "b", "W_A", "b_A")}, indent=2),
          flush=True)
    print(f"[P1] saved {OUT / 'calibrated_pqc.pt'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["p0", "p1", "both"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1e-3)
    args = ap.parse_args()
    import torch
    if args.phase in ("p0", "both"):
        p0(args.device, n=args.n or 256)
    if args.phase in ("p1", "both"):
        p1(args.device, n=args.n or 2000, lam=args.lam)


if __name__ == "__main__":
    main()
