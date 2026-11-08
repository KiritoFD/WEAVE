#!/usr/bin/env bash
# Decode/generation throughput vs batch size (images only, no metric models).
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
CK="runs/blog_sbflow/Mid8M_noOT/model_step1000.pt runs/blog_sbflow/Mid8M_noOT/model_step1500.pt"

for B in 4 8 16 32; do
  OUT=runs/blog_sbflow/gen/_bsweep_b$B
  rm -rf "$OUT"
  echo "=== batch=$B ==="
  python3 -u -m sbflow.generate --repo-root . --ckpt $CK \
    --style-a Impressionism --style-b Ukiyo_e \
    --out "$OUT" --steps 4 --method heun \
    --batch "$B" --workers 4 --save-images 2>&1 | grep -E "^  \[|one-time"
done
