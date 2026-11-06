#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== bench2 log =="
cat _tmp/bench_batch2.log
echo "== proc =="
ps aux | grep -E "bench_batch2|pgt_render" | grep -v grep | head -3
ls -l runs/v4/b64_t8.log 2>/dev/null && tail -n 20 runs/v4/b64_t8.log
