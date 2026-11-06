#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
nohup bash _tmp/bench_batch2.sh > _tmp/bench_batch2.log 2>&1 &
echo "pid=$!"
