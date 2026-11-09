#!/usr/bin/env bash
# Wait for the current core tmux session to finish, then continue the remaining 31 arms.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
mkdir -p "$ROOT/logs"
echo "[continuation] $(date +%H:%M:%S) waiting for tmux session 'suite40' to end"
while tmux has-session -t suite40 2>/dev/null; do
  sleep 300
done
echo "[continuation] $(date +%H:%M:%S) core session ended; starting rest"
python3 -u -m sbflow.suite40 --root "$ROOT" --arms rest --resume-suite --max-hours 7.5 \
  2>&1 | tee -a "$ROOT/logs/suite_rest.log"
echo "[continuation] $(date +%H:%M:%S) DONE"
