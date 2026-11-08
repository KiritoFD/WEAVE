#!/usr/bin/env bash
# Evaluate one two-domain checkpoint on its frozen Test600 subset (60 ordered tasks).
# Args: TAG  CKPT  STEPS  METHOD  WEIGHTS
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
TAG=${1:?tag}
CKPT=${2:?ckpt}
STEPS=${3:-4}
METHOD=${4:-heun}
WEIGHTS=${5:-raw}
EVAL=runs/blog_sbflow/eval_${TAG}

echo "=== generate ${TAG} (${METHOD} ${STEPS} steps, ${WEIGHTS}) ==="
python3 -u -m sbflow.eval_test600 generate \
  --repo-root . --ckpt "$CKPT" \
  --style-a Impressionism --style-b Ukiyo_e \
  --output "$EVAL" --weights "$WEIGHTS" \
  --method "$METHOD" --steps "$STEPS" --batch 2

echo "=== score ${TAG} ==="
python3 -u -m sbflow.eval_test600 score \
  --repo-root . --manifest "$EVAL/manifest.json" \
  --output "$EVAL/style_scores.json" --batch 4

echo "=== EVAL DONE ${TAG} ==="
