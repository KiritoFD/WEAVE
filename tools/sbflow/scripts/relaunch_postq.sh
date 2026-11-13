#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
tmux kill-session -t postq 2>/dev/null && echo "killed postq" || echo "postq not present"
sleep 2
tmux new-session -d -s postq -n q \
  "bash -u sbflow/scripts/postqueue_v6_24h.sh; echo POSTQUEUE_EXIT=\$? >> runs/v6_24h/postqueue.log"
sleep 3
echo "=== sessions ==="
tmux ls
echo "=== postq log tail ==="
tail -n 3 runs/v6_24h/postqueue.log 2>/dev/null || echo "  none"
