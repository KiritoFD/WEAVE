#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== queue_p2c.log (full, last 30) ==="
tail -n 30 runs/v6_24h/queue_p2c.log 2>/dev/null || echo "  no log"
echo
echo "=== all python train processes ==="
ps -eo pid,etime,args --no-headers | grep train_eval_v6 | grep -v grep
echo
echo "=== now ==="
date +%H:%M:%S
echo
echo "=== log mtime ==="
ls -l --time-style=+%H:%M:%S runs/v6_24h/queue_p2c.log 2>/dev/null
