#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
setsid -f bash _tmp/bench_batch2.sh > /dev/null 2>&1 < /dev/null
sleep 8
ps aux | grep -E "bench_batch2|pgt_render" | grep -v grep | head -3
nvidia-smi --query-gpu=memory.used --format=csv,noheader
