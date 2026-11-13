#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "now: $(date +%H:%M:%S)"
echo "=== current arm ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -n 1 | cut -c1-90
echo "=== last step lines ==="
for a in N7_dim24 N7_dim40; do
  f="runs/v6_24h/train_logs/${a}.log"
  [ -f "$f" ] || continue
  echo "-- $a"
  grep -a "step " "$f" | tail -n 3
done
echo "=== sessions ==="
tmux ls
