#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import sys, torch
sys.path.insert(0, ".")
from tools.v6.models_v6 import C3UNet
target = 1881792
print("base_dim -> params (mode=direct)")
best = None
for d in (16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56):
    m = C3UNet(base_dim=d, num_styles=5, mode="direct")
    n = sum(p.numel() for p in m.parameters())
    mark = ""
    if best is None or abs(n - target) < abs(best[1] - target):
        best = (d, n)
    print(f"  {d:3d} -> {n:9d} {mark}")
print(f"\nclosest to D1_c1_gentle ({target}): base_dim={best[0]} -> {best[1]} "
      f"(diff {best[1]-target:+d})")
PY
