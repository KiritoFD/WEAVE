#!/usr/bin/env bash
# Speed test for image production only (no metric models loaded).
# Runs 4 checkpoints through one session so the cached source encoding is visible.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
OUT=${1:-runs/blog_sbflow/gen/Mid8M}
BATCH=${2:-8}
WORKERS=${3:-4}
rm -rf "$OUT"
echo "=== generate 4 checkpoints, batch=$BATCH workers=$WORKERS ==="
python3 -u -m sbflow.generate --repo-root . \
  --ckpt runs/blog_sbflow/Mid8M_noOT/model_step500.pt \
         runs/blog_sbflow/Mid8M_noOT/model_step1000.pt \
         runs/blog_sbflow/Mid8M_noOT/model_step1500.pt \
         runs/blog_sbflow/Mid8M_noOT/model_step2000.pt \
  --style-a Impressionism --style-b Ukiyo_e \
  --out "$OUT" --steps 4 --method heun \
  --batch "$BATCH" --workers "$WORKERS" \
  --save-images --save-noop
echo "=== DONE ==="
