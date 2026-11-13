#!/usr/bin/env bash
# Build DCAE caches for all five styles from the canonical 15,545-image union (see data.md).
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
AE=/mnt/i/Latent_Style/weights/dc-ae-f32c32-mix-1.0-diffusers
OUT=runs/blog_sbflow/cache5
mkdir -p "$OUT"

for S in Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e; do
  if [ -f "$OUT/$S.pt" ]; then echo "skip existing $S"; continue; fi
  echo "=== caching $S ==="
  python3 -u -m sbflow.cache \
    --style "$S" \
    --images "datasets/wikiart_new_512/$S" "datasets/distinct5_512/train/$S" \
    --out "$OUT/$S.pt" --ae "$AE" \
    --image-size 512 --batch 8 --device cuda \
    --no-dedupe
done
echo "=== CACHE5 DONE ==="
ls -lh "$OUT"
