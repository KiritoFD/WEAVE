#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "=== queue log ==="; tail -8 logs/v8_queue.log
echo "=== train log ==="; tail -8 logs/v8_train.log
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== tmux ==="; tmux ls 2>/dev/null
