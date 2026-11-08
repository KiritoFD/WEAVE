#!/usr/bin/env bash
# Cache DCAE latents for the first two-domain blog reproduction (Impressionism <-> Ukiyo_e).
# RGB thumbnails only: WSL has no network, so the optional DINO cache is disabled here.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
AE=/mnt/i/Latent_Style/weights/dc-ae-f32c32-mix-1.0-diffusers
OUT=runs/blog_sbflow/cache
mkdir -p "$OUT"

for STYLE in Impressionism Ukiyo_e; do
  if [ -f "$OUT/${STYLE}.pt" ]; then
    echo "skip existing $OUT/${STYLE}.pt"
    continue
  fi
  echo "=== caching $STYLE ==="
  python3 -u -m sbflow.cache \
    --images "datasets/distinct5_512/train/$STYLE" \
    --out "$OUT/${STYLE}.pt" \
    --ae "$AE" \
    --image-size 512 --batch 8 --device cuda
done
echo "=== CACHE DONE ==="
ls -lh "$OUT"
