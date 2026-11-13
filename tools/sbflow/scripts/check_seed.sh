#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
for a in ("P2_G0_bf16", "Q1_G0_s43", "P2_G1b_bf16"):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        print(f"  {a:14s} pending"); continue
    d = json.loads(p.read_text())
    e = d.get("eval200") or {}
    print(f"  {a:14s} seed={d.get('seed')} dT={e.get('mean_dT'):+.6f} "
          f"Top1={sum((e.get('per_target_top1') or {}).values())/max(1,len(e.get('per_target_top1') or {})):.6f}")
PY
