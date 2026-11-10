#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import torch, numpy as np
from pathlib import Path
ASSET = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
d = torch.load(ASSET / "00_Early_Renaissance.pt", map_location="cpu", weights_only=False)
lat = d["latents"].float()
print("keys:", list(d.keys()))
print("shape:", tuple(lat.shape), "dtype:", lat.dtype)
print(f"per-sample squared norm: mean={float(lat.flatten(1).pow(2).sum(1).mean()):.4f}")
print(f"per-dim RMS: {float(lat.pow(2).mean().sqrt()):.6f}")
print(f"abs mean: {float(lat.abs().mean()):.6f}  min={float(lat.min()):.4f} max={float(lat.max()):.4f}")
D = lat[0].numel()
print(f"dims per sample: {D}")
print(f"if a 0.25 per-dim step were applied: ||d||^2 = {0.0625 * D:.1f}")
PY
