#!/usr/bin/env bash
# Wait until all 40 arms have a terminal status, then run Full750 for the three best
# distinct mechanisms plus exp01. Decoupled from tmux session lifetime.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
mkdir -p "$ROOT/logs"
echo "[full750-watch] $(date +%H:%M:%S) waiting for all 40 arms"
while true; do
  if python3 - <<'PY'
import json, sys
from pathlib import Path
p = Path("runs/blog_sbflow/suite40/suite_state.json")
s = json.loads(p.read_text()) if p.exists() else {}
terminal = {"done", "failed", "incomplete", "trained"}
missing = [f"{i:02d}" for i in range(1, 41)
           if f"{i:02d}" not in s or s[f"{i:02d}"].get("status") not in terminal]
if missing:
    print(f"  waiting for {len(missing)} arms: {missing[:8]}", flush=True)
    sys.exit(1)
print("  all 40 arms terminal", flush=True)
PY
  then
    break
  fi
  sleep 600
done
echo "[full750-watch] $(date +%H:%M:%S) selecting finalists"
python3 -u -m sbflow.full750 --score-threads 4 2>&1 | tee -a "$ROOT/logs/full750.log"
echo "[full750-watch] $(date +%H:%M:%S) DONE"
