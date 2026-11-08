#!/usr/bin/env bash
# 8M model (width256 depth6 heads4): VRAM and speed at large microbatches.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 -u -m sbflow.scripts.probe_batch \
  --tag v9mid_probe_large --width 256 --depth 6 --heads 4 --patch 2 \
  --batches 512,1024,1600,2048 --reps 6 --warmup 2
