#!/usr/bin/env bash
# After all 40 arms AND Full750 are done, evaluate the 1000/2000/3000/4000
# equivalent-step checkpoints to detect V9-style late degradation.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
mkdir -p "$ROOT/logs"
echo "[trend-watch] $(date +%H:%M:%S) waiting for 40 arms + Full750"
while true; do
  if python3 - <<'PY'
import json, sys
from pathlib import Path
r = Path("runs/blog_sbflow/suite40")
s = json.loads((r/"suite_state.json").read_text()) if (r/"suite_state.json").exists() else {}
terminal = {"done", "failed", "incomplete", "trained"}
if any(f"{i:02d}" not in s or s[f"{i:02d}"].get("status") not in terminal
       for i in range(1, 41)):
    print("  arms not all terminal yet", flush=True)
    sys.exit(1)
if not (r/"full750_results.json").exists():
    print("  Full750 not written yet", flush=True)
    sys.exit(1)
print("  all ready", flush=True)
PY
  then
    break
  fi
  sleep 900
done
echo "[trend-watch] $(date +%H:%M:%S) running checkpoint trend"
python3 -u -m sbflow.checkpoint_trend40 --arms 01 --threads 4 \
  2>&1 | tee -a "$ROOT/logs/checkpoint_trend.log"
echo "[trend-watch] $(date +%H:%M:%S) DONE"
