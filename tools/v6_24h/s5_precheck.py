"""V6 24h plan section 6 - S5 pre-experiment (<=30 min, CPU only).

Before adding an HF structure loss we must show it carries information the existing
losses do NOT already have. The plan requires exactly this check and says to CANCEL S5 if
the statistics are near-redundant or cannot separate the hard targets.

What is measured, per style, from real TRAINING latents:
  * directional neighbour correlation of the Haar HF bands at shifts 1/2/4 px,
    horizontally and vertically (long-range brush-stroke directionality);
  * second-order directional co-occurrence (corr between |LH| and |HL|, sign agreement).

Two questions:
  Q1 does this feature separate the five styles at all? (5-way nearest-centroid,
     leave-one-out; chance = 20%)
  Q2 is it redundant with the existing DWT subband moments used by V6? (correlation of the
     two style-distance matrices)

Verdict rule (pre-declared):
  keep S5 only if Q1 accuracy >= 40% AND Q2 distance-matrix correlation <= 0.8.
  Otherwise cancel S5 and record it in Negative_Results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from wavelet import dwt2_haar

CACHE = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
HARD = ["Rococo", "Early_Renaissance", "Ukiyo_e"]


def hf_features(z: torch.Tensor) -> np.ndarray:
    """Directional HF statistics for one latent (1,4,64,64) -> fixed-length vector."""
    _, lh, hl, hh = dwt2_haar(z.float())
    feats = []
    for name, b in (("lh", lh), ("hl", hl), ("hh", hh)):
        b = b[0]                                     # (4,H,W)
        b = b - b.mean(dim=(1, 2), keepdim=True)
        s = b.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        bn = b / s
        for d in (1, 2, 4):
            # horizontal neighbour correlation
            a1 = bn[:, :, :-d].reshape(4, -1)
            a2 = bn[:, :, d:].reshape(4, -1)
            ch = (a1 * a2).mean(dim=1)
            # vertical neighbour correlation
            b1 = bn[:, :-d, :].reshape(4, -1)
            b2 = bn[:, d:, :].reshape(4, -1)
            cv = (b1 * b2).mean(dim=1)
            feats += [ch.mean().item(), cv.mean().item()]
        # amplitude of the band relative to LL (texture energy share)
        feats.append(float(b.abs().mean()))
    # second-order: LH-HL amplitude co-movement
    al = lh[0].abs().mean(dim=(1, 2))
    ah = hl[0].abs().mean(dim=(1, 2))
    feats.append(float((al * ah).mean() / (al.mean() * ah.mean() + 1e-8)))
    # sign agreement between LH and HL (directional coherence)
    sg = (torch.sign(lh[0]) * torch.sign(hl[0])).mean()
    feats.append(float(sg))
    return np.asarray(feats, dtype=np.float32)


def dwt_moment_features(z: torch.Tensor) -> np.ndarray:
    """The statistics V6 already uses: per-band mean/std plus cross-band covariance."""
    ll, lh, hl, hh = dwt2_haar(z.float())
    out = []
    for b in (ll, lh, hl, hh):
        out += [b.mean().item(), b.std().item()]
    flat = torch.cat([lh[0], hl[0], hh[0]], 0).reshape(12, -1)
    cov = (flat @ flat.t()) / flat.shape[1]
    iu = torch.triu_indices(12, 12, offset=1)
    out += cov[iu[0], iu[1]].tolist()
    return np.asarray(out, dtype=np.float32)


def loo_nearest_centroid(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out 5-way nearest-centroid accuracy."""
    correct = 0
    for i in range(len(X)):
        mask = np.arange(len(X)) != i
        cents = np.stack([X[mask & (y == c)].mean(0) for c in range(len(STYLES))])
        d = ((cents - X[i]) ** 2).sum(1)
        correct += int(d.argmin() == y[i])
    return correct / len(X)


def dist_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    cents = np.stack([X[y == c].mean(0) for c in range(len(STYLES))])
    D = np.zeros((len(STYLES), len(STYLES)))
    for i in range(len(STYLES)):
        for j in range(len(STYLES)):
            D[i, j] = np.linalg.norm(cents[i] - cents[j])
    return D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=150)
    ap.add_argument("--out", default="runs/v6_24h/s5_precheck.json")
    args = ap.parse_args()

    X_hf, X_mo, y = [], [], []
    for si, (f, s) in enumerate(zip(FILES, STYLES)):
        d = torch.load(CACHE / f, map_location="cpu", weights_only=False)
        lat = d["latents"].float()
        idx = np.linspace(0, len(lat) - 1, min(args.per_style, len(lat))).astype(int)
        for i in idx:
            z = lat[i:i + 1]
            X_hf.append(hf_features(z))
            X_mo.append(dwt_moment_features(z))
            y.append(si)
        print(f"  {s:20s} n={len(idx)}", flush=True)
    X_hf = np.stack(X_hf)
    X_mo = np.stack(X_mo)
    y = np.asarray(y)

    # standardize per dimension (fit on all samples; this is a discriminability probe,
    # not a classifier we ship)
    def std(A):
        return (A - A.mean(0)) / (A.std(0) + 1e-6)
    acc_hf = loo_nearest_centroid(std(X_hf), y)
    acc_mo = loo_nearest_centroid(std(X_mo), y)
    print(f"\nQ1 5-way LOO nearest-centroid accuracy (chance 0.20)")
    print(f"  HF directional features : {acc_hf:.3f}")
    print(f"  existing DWT moments    : {acc_mo:.3f}")

    D_hf = dist_matrix(std(X_hf), y)
    D_mo = dist_matrix(std(X_mo), y)
    iu = np.triu_indices(len(STYLES), 1)
    # correlation of the two distance matrices (Mantel-style, no permutation needed for
    # a coarse redundancy read)
    q2 = float(np.corrcoef(D_hf[iu], D_mo[iu])[0, 1])
    print(f"\nQ2 distance-matrix correlation (HF vs DWT moments) = {q2:.3f}")
    print("  HF pairwise style distances:")
    for i in range(len(STYLES)):
        print("   ", f"{STYLES[i]:20s}", " ".join(f"{D_hf[i,j]:6.2f}" for j in range(len(STYLES))))

    # hard-target separation: does HF put the three hard styles farther apart than the
    # existing moments already do?
    def hard_sep(D):
        vals = [D[STYLES.index(a), STYLES.index(b)]
                for i, a in enumerate(HARD) for b in HARD[i + 1:]]
        return float(np.mean(vals))
    hs_hf, hs_mo = hard_sep(D_hf), hard_sep(D_mo)
    print(f"\n  hard-target mean pairwise distance: HF {hs_hf:.2f} vs moments {hs_mo:.2f}")

    keep = (acc_hf >= 0.40) and (q2 <= 0.80)
    print(f"\nVERDICT: {'KEEP S5' if keep else 'CANCEL S5'}")
    print(f"  rule: keep iff HF accuracy >= 0.40 AND distance-matrix corr <= 0.80")
    print(f"  actual: accuracy {acc_hf:.3f}, corr {q2:.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_per_style": args.per_style,
        "acc_hf": acc_hf, "acc_moments": acc_mo,
        "dist_matrix_corr": q2,
        "dist_hf": D_hf.tolist(), "dist_moments": D_mo.tolist(),
        "hard_sep_hf": hs_hf, "hard_sep_moments": hs_mo,
        "verdict": "keep" if keep else "cancel",
        "rule": "keep iff acc_hf >= 0.40 and dist_matrix_corr <= 0.80",
        "styles": STYLES,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
