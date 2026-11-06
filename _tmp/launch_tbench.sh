#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
setsid -f bash _tmp/bench_train.sh > /dev/null 2>&1 < /dev/null
sleep 10
ps aux | grep -E "run.py" | grep -v grep | head -2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
