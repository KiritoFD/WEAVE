#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== S6 header ==="
head -n 12 runs/v6_24h/train_logs/S6_style_keys.log | grep -aE "params=|B1 branch|reference pool|style_keys"
echo
echo "=== style-key params present in checkpoint-in-progress? (check via model build) ==="
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import torch, sys
sys.path.insert(0, ".")
from tools.v6.models_v6 import C3UNet
for sk in (False, True):
    m = C3UNet(base_dim=32, num_styles=5, mode="direct", style_keys=sk)
    n = sum(p.numel() for p in m.parameters())
    nsk = sum(p.numel() for p in m.skr.parameters()) if m.skr is not None else 0
    print(f"  style_keys={sk!s:5s} total={n:,}  skr={nsk:,}  has_skr={m.skr is not None}")
PY
