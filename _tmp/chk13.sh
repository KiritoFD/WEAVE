#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
ls -l _tmp/bench_batch2.sh _tmp/launch_bench2.sh
head -3 _tmp/bench_batch2.sh | cat -A | head -3
bash _tmp/bench_batch2.sh > /tmp/bb2.log 2>&1 &
echo "pid=$!"
sleep 20
tail -n 5 /tmp/bb2.log
ls -l runs/v4/batch_bench2.txt
nvidia-smi --query-gpu=memory.used --format=csv,noheader
