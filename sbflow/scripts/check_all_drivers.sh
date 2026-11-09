#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== sessions ==="
tmux ls
echo
echo "=== postq log (last 8) ==="
tail -n 8 runs/v6_24h/postqueue.log 2>/dev/null || echo "  none"
echo
echo "=== pass3 log (last 6) ==="
tail -n 6 runs/v6_24h/pass3.log 2>/dev/null || echo "  none"
echo
echo "=== current training arm ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -n 1 | cut -c1-80
