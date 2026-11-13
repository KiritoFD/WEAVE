#!/usr/bin/env bash
# Restart the three watcher sessions with the race-free scripts.
# Never touches the main 'suite40' training session.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
for s in suite40_continue suite40_full750 suite40_trend; do
  tmux kill-session -t "$s" 2>/dev/null || true
done
sleep 2
bash -u sbflow/scripts/start_continuation_tmux.sh
bash -u sbflow/scripts/start_full750_tmux.sh
bash -u sbflow/scripts/start_trend_tmux.sh
echo "=== sessions ==="
tmux ls
