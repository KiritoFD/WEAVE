#!/usr/bin/env bash
# Wait until the nine core arms have terminal status, then run the remaining 31.
# Do NOT wait for the tmux session itself: the core launcher intentionally keeps its
# session alive after the driver exits, so `tmux has-session` never becomes false.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
mkdir -p "$ROOT/logs"
CORE="01 04 08 10 24 27 32 35 37"
echo "[continuation] $(date +%H:%M:%S) waiting for core arms to finish"
while true; do
  # Never start a second driver while the core driver (or any other suite40 driver)
  # is still alive: they would both rewrite suite_state.json and lose results.
  if pgrep -f "python3 -u -m sbflow.suite40" >/dev/null 2>&1; then
    echo "  a suite40 driver is still running; deferring rest" >&2
    sleep 300
    continue
  fi
  if python3 - <<'PY'
import json, sys
from pathlib import Path
p = Path("runs/blog_sbflow/suite40/suite_state.json")
s = json.loads(p.read_text()) if p.exists() else {}
core = ["01", "04", "08", "10", "24", "27", "32", "35", "37"]
terminal = {"done", "failed", "incomplete", "trained", "awaiting_score"}
pending = [c for c in core
           if c not in s or s[c].get("status") not in terminal]
if pending:
    print(f"  core still running: {pending}", flush=True)
    sys.exit(1)
print("  all core arms terminal", flush=True)
PY
  then
    break
  fi
  sleep 300
done
echo "[continuation] $(date +%H:%M:%S) core done; starting rest (31 arms)"
python3 -u -m sbflow.suite40 --root "$ROOT" --arms rest --resume-suite --max-hours 7.5 \
  2>&1 | tee -a "$ROOT/logs/suite_rest.log"
echo "[continuation] $(date +%H:%M:%S) DONE"
