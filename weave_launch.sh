#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
mkdir -p runs/v2/logs runs/v2/eval
setsid nohup bash weave_run_v2_v3.sh > runs/v2/run_v2_v3_wrapper.log 2>&1 < /dev/null &
echo $! > runs/v2/run_v2_v3.pid
