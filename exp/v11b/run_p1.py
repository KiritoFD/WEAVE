"""V11-B P1: S10 high-frequency loss audit on held-out real paintings.

For N holdout paintings: encode with SD0 VAE -> S10 decode -> compare
2-level Haar band statistics (energy, per-band cosine) between the
original and the reconstruction. Quantifies how much painting detail
the VAE round-trip already destroys — if small, the wavelet flow is
still worth testing but we cannot claim the VAE is the bottleneck.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .bands import dwt2
from .dataset_real import RealWaveletDataset

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "v11b"
SD0_VAE = REPO / "deliver" / "checkpoints" / "sd0_vae"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"

    from diffusers import AutoencoderKL
    from src.utils import decode_s10, load_s10_decoder
    vae = AutoencoderKL.from_pretrained(str(SD0_VAE), torch_dtype=torch.float16,
                                      local_files_only=True).to(device).eval()
    s10 = load_s10_decoder(device)

    ds = RealWaveletDataset("holdout", crop=512)
    g = torch.Generator().manual_seed(7)
    idx = torch.randperm(len(ds), generator=g)[:args.n].tolist()
    items = [ds[i] for i in idx]

    stats = {"H1": [], "H2": [], "cos_H1": [], "cos_H2": []}
    sf = 0.18215
    for c0 in range(0, len(items), args.batch):
        batch = items[c0:c0 + args.batch]
        x = torch.stack([b[0] for b in batch]).to(device)  # [B,3,512,512] 0-1
        x_in = x * 2 - 1
        with torch.autocast("cuda", dtype=torch.float16):
            z = vae.encode(x_in.half()).latent_dist.mean * sf
        rec = decode_s10(s10, z.to(device), batch=args.batch)  # [0,1]
        rec = rec.to(device)
        d0 = dwt2(x)
        d1 = dwt2(rec)
        for band in ("H1", "H2"):
            a, b = d0[band].float(), d1[band].float()
            ea = a.flatten(1).pow(2).mean(1)
            eb = b.flatten(1).pow(2).mean(1)
            stats[band].append((eb / ea.clamp_min(1e-12)).cpu())
            ca = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
            stats[f"cos_{band}"].append(ca.cpu())
        print(f"  {min(c0+args.batch,len(items))}/{len(items)}", flush=True)

    out = {}
    for k, v in stats.items():
        t = torch.cat(v)
        out[k] = {"mean": float(t.mean()), "median": float(t.median()),
                  "p10": float(t.quantile(0.1)), "p90": float(t.quantile(0.9))}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "P1_S10_HF_AUDIT.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
