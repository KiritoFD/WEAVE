#!/usr/bin/env bash
# V9 blog-faithful two-domain run. Args: TAG  OT_RATIO  STEPS  EPS
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
TAG=${1:?tag}
OT=${2:?ot_ratio}
STEPS=${3:-8000}
EPS=${4:-0.04}
OUT=runs/blog_sbflow/${TAG}
mkdir -p "$OUT"
echo "=== training $TAG ot_ratio=$OT steps=$STEPS eps=$EPS ==="
python3 -u -m sbflow.train \
  --source-cache runs/blog_sbflow/cache/Impressionism.pt \
  --target-cache runs/blog_sbflow/cache/Ukiyo_e.pt \
  --output "$OUT" \
  --steps "$STEPS" --microbatch 16 --accum 4 \
  --eps "$EPS" --ot-ratio "$OT" --ot-reg 0.05 \
  --width 384 --depth 12 --heads 6 --patch 2 \
  --lr 2e-4 --min-t 0.002 \
  --log-interval 200 --save-interval 2000 \
  --moments-batches 64
echo "=== DONE $TAG ==="
ls -lh "$OUT"
