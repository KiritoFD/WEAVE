#!/usr/bin/env bash
# Calibrate microbatch for V9-Lite so training actually fills the GPU (~10 GB).
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 -u -m sbflow.scripts.probe_batch \
  --tag v9lite_probe --width 128 --depth 6 --heads 4 --patch 2 \
  --batches 32,64,128,256,384,512 --reps 12 --warmup 4
