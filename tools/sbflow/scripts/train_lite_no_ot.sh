#!/usr/bin/env bash
# V9-Lite training with OT disabled (no Sinkhorn) — cost-focused run.
# Args: TAG  STEPS  MICROBATCH  ACCUM
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
TAG=${1:?tag}; STEPS=${2:-2000}; MB=${3:-256}; ACC=${4:-1}
OUT=runs/blog_sbflow/${TAG}
mkdir -p "$OUT"
echo "=== V9-Lite ${TAG}: steps=$STEPS mb=$MB accum=$ACC ot=OFF warmup=100 ==="
python3 -u -m sbflow.train \
  --source-cache runs/blog_sbflow/cache/Impressionism.pt \
  --target-cache runs/blog_sbflow/cache/Ukiyo_e.pt \
  --output "$OUT" \
  --steps "$STEPS" --microbatch "$MB" --accum "$ACC" \
  --eps 0.04 --ot-ratio 0.0 --ot-reg 0.05 \
  --width 128 --depth 6 --heads 4 --patch 2 \
  --lr 2e-4 --min-t 0.002 --warmup 100 \
  --log-interval 100 --save-interval 500 \
  --moments-batches 32
echo "=== DONE ${TAG} ==="
ls -lh "$OUT"
