#!/usr/bin/env bash
# NFE sweep for one checkpoint on its Test600 pair subset.
# Args: TAG  CKPT
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
TAG=${1:?tag}
CKPT=${2:?ckpt}

for CFG in "heun 2" "heun 8" "euler 4"; do
  set -- $CFG
  METHOD=$1; STEPS=$2
  EVAL=runs/blog_sbflow/eval_${TAG}_${METHOD}${STEPS}
  if [ -f "$EVAL/style_scores.json" ]; then
    echo "skip existing $EVAL"
    continue
  fi
  echo "=== ${TAG} ${METHOD}${STEPS} ==="
  python3 -u -m sbflow.eval_test600 generate \
    --repo-root . --ckpt "$CKPT" \
    --style-a Impressionism --style-b Ukiyo_e \
    --output "$EVAL" --weights raw \
    --method "$METHOD" --steps "$STEPS" --batch 2
  python3 -u -m sbflow.eval_test600 score \
    --repo-root . --manifest "$EVAL/manifest.json" \
    --output "$EVAL/style_scores.json" --batch 4
done
echo "=== NFE SWEEP DONE ${TAG} ==="
