#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== now ==="
date +%H:%M:%S
echo "=== L2 training steps ==="
grep -a "step " runs/v6_24h/train_logs/L2_highS.log | tail -n 3
echo "=== L2 process ==="
ps -eo etime,args --no-headers | grep "L2_highS" | grep -v grep | head -n 1
echo "=== queue log tail ==="
tail -n 3 runs/v6_24h/queue.log
