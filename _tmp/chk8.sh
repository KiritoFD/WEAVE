#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== bench results =="
cat runs/v4/batch_bench.txt 2>/dev/null
echo "== current =="
ps aux | grep -E "pgt_render" | grep -v grep | head -2
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "== last run tail =="
ls -t runs/v4/bench_b*.log 2>/dev/null | head -1 | xargs -r tail -n 4
