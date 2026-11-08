#!/usr/bin/env bash
# Time a fixed number of optimizer updates to budget the real run.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
N=${1:-40}
rm -rf runs/blog_sbflow/_bench
/usr/bin/time -f "WALL_SEC=%e PEAK_KB=%M" python3 -u -m sbflow.train \
  --source-cache runs/blog_sbflow/cache/Impressionism.pt \
  --target-cache runs/blog_sbflow/cache/Ukiyo_e.pt \
  --output runs/blog_sbflow/_bench \
  --steps "$N" --microbatch 16 --accum 4 \
  --eps 0.04 --ot-ratio 0.5 --ot-reg 0.05 \
  --width 384 --depth 12 --heads 6 --patch 2 \
  --lr 2e-4 --min-t 0.002 --log-interval 10 --save-interval "$N" \
  --moments-batches 8 2>&1 | tail -n 8
