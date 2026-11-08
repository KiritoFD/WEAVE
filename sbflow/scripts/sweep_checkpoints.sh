#!/usr/bin/env bash
# Evaluate every saved checkpoint of one run to map the style-vs-training curve.
# Args: RUNNAME  (checkpoints model_step*.pt in runs/blog_sbflow/$RUNNAME)
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
RUN=${1:?run}
STEPS=${2:-4}
METHOD=${3:-heun}

for CK in runs/blog_sbflow/$RUN/model_step*.pt; do
  [ -f "$CK" ] || continue
  STEP=$(basename "$CK" .pt | sed 's/model_step//')
  TAG="${RUN}_step${STEP}"
  EVAL=runs/blog_sbflow/eval_${TAG}
  if [ -f "$EVAL/style_scores.json" ]; then echo "skip $TAG"; continue; fi
  echo "=== $TAG ==="
  python3 -u -m sbflow.eval_test600 generate \
    --repo-root . --ckpt "$CK" --style-a Impressionism --style-b Ukiyo_e \
    --output "$EVAL" --weights raw --method "$METHOD" --steps "$STEPS" --batch 2
  python3 -u -m sbflow.eval_test600 score \
    --repo-root . --manifest "$EVAL/manifest.json" \
    --output "$EVAL/style_scores.json" --batch 4 > /dev/null
  python3 -c "
import json;d=json.load(open('$EVAL/style_scores.json'));
g=d['aggregate']['generated']['cross_overall'];n=d['aggregate']['dcae_noop']['cross_overall']
print('$TAG', 'top1=%.4f'%g['target_top1'], 'margin=%.4f'%g['target_margin'], 'dT=%.4f'%g['csd_dT'], 'lpips=%.4f'%g['lpips_to_input'], '| noop top1=%.4f'%n['target_top1'])
"
done
echo "=== CHECKPOINT SWEEP DONE $RUN ==="
