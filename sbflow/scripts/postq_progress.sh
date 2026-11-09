#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== postq progress ==="
grep -aE "^##########|STEP FAILED|SMOKE DONE|^\[[0-9]{2}:" runs/v6_24h/postqueue.log | tail -n 14
echo
echo "=== sessions ==="
tmux ls
echo
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
