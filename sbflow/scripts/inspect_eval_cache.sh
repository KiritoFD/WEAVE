#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== eval_protocol/cache ==="
ls -lh eval_protocol/cache/ 2>/dev/null | head -20
echo "=== keys of candidate caches ==="
python3 - <<'PY'
import torch, glob
from pathlib import Path
for f in sorted(glob.glob("eval_protocol/cache/*.pt")):
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(d, dict):
            ks = list(d.keys())
            print(f"  {Path(f).name}: {ks[:8]}")
            if "styles" in d and isinstance(d["styles"], dict):
                s0 = next(iter(d["styles"]))
                v = d["styles"][s0]
                print(f"      styles[{s0}] keys={list(v.keys()) if isinstance(v, dict) else type(v)}")
    except Exception as e:
        print(f"  {Path(f).name}: ERR {e}")
PY
