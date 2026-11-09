#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
tmux kill-session -t pass3 2>/dev/null && echo "killed pass3" || echo "pass3 not present"
sleep 2
tmux new-session -d -s pass3 -n q \
  "bash -u sbflow/scripts/pass3_v6_24h.sh; echo PASS3_EXIT=\$? >> runs/v6_24h/pass3.log"
sleep 3
echo "=== sessions ==="
tmux ls
echo "=== pass3 log ==="
cat runs/v6_24h/pass3.log 2>/dev/null || echo "  (no log yet)"
