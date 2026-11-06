#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== bench2 =="
cat runs/v4/batch_bench2.txt
echo "== log =="
tail -n 8 /tmp/bb2.log
echo "== proc =="
ps aux | grep -E "bench_batch2|pgt_render" | grep -v grep | head -3
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
