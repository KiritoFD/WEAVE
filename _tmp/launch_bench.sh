#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
nohup bash _tmp/bench_batch.sh > _tmp/bench_batch.log 2>&1 &
echo "pid=$!"
sleep 5
tail -n 3 _tmp/bench_batch.log
