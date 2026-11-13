#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== now ==="
date +%H:%M:%S
echo "=== L5 log tail ==="
grep -a "step " runs/v6_24h/train_logs/L5_same_off.log | tail -n 2
echo "=== L5 process ==="
ps -eo etime,args --no-headers | grep "L5_same_off" | grep -v grep | head -n 1
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
