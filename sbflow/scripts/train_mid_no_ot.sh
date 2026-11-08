#!/usr/bin/env bash
# V9 mid-size model (7.45M) with OT disabled, same batch/steps as V9-Lite.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
TAG=${1:-Mid8M_noOT}; STEPS=${2:-2000}; MB=${3:-256}; ACC=${4:-1}
OUT=runs/blog_sbflow/${TAG}
mkdir -p "$OUT"
echo "=== V9-Mid ${TAG}: width256 depth6 heads4 steps=$STEPS mb=$MB accum=$ACC ot=OFF ==="
python3 -u -m sbflow.train \
  --source-cache runs/blog_sbflow/cache/Impressionism.pt \
  --target-cache runs/blog_sbflow/cache/Ukiyo_e.pt \
  --output "$OUT" \
  --steps "$STEPS" --microbatch "$MB" --accum "$ACC" \
  --eps 0.04 --ot-ratio 0.0 --ot-reg 0.05 \
  --width 256 --depth 6 --heads 4 --patch 2 \
  --lr 2e-4 --min-t 0.002 --warmup 100 \
  --log-interval 100 --save-interval 500 \
  --moments-batches 32
echo "=== DONE ${TAG} ==="
ls -lh "$OUT"
