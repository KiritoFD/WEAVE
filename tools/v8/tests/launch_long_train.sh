#!/usr/bin/env bash
set -euo pipefail
mkdir -p /mnt/i/Latent_Style/WEAVE/runs/v8/logs
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
setsid nohup python3 -u tools/v8/run_long_training.py \
  --steps 8000 --which B0,B2 --checkpoints 800,3000,8000 \
  > runs/v8/logs/long_train.out 2>&1 < /dev/null &
echo "launched pid $!"
