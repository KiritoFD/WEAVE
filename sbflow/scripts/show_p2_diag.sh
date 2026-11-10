#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== G1 step-1 gradient ratio (the calibration evidence) ==="
grep -ah "G1 diag step1" runs/v6_24h/train_logs/*.log runs/v6/G1*_smoke/*.log 2>/dev/null | tail -n 6
for a in G0_smoke G1a_smoke G1b_smoke; do
  f="runs/v6_24h/train_logs/${a}.log"
  [ -f "$f" ] || continue
  echo "-- $a"
  grep -a "G1 diag step1" "$f" || echo "   (no rank loss)"
  grep -a "init_from=" "$f" | head -n 1
done
echo
echo "=== smoke reports ==="
python3 - <<'PY'
import json
from pathlib import Path
for a in ("G0_smoke", "G1a_smoke", "G1b_smoke"):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        print(f"  {a}: no report"); continue
    d = json.loads(p.read_text())
    e = d.get("eval200") or {}
    print(f"  {a:10s} dT={e.get('mean_dT',0):+.4f} dM={e.get('mean_target_margin',0):+.4f} "
          f"DINO={e.get('mean_dino_c',0):.4f} LPIPS={e.get('mean_lpips',0):.4f} "
          f"sty+={e.get('styles_improved')}")
PY
