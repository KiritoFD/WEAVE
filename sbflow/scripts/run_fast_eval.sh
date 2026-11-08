#!/usr/bin/env bash
# Fast eval parity + speed check against the known eval_test600 result.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
OUT=${1:-runs/blog_sbflow/fast_eval/parity}
shift || true
echo "=== fast eval: 4 Mid8M checkpoints in ONE process ==="
python3 -u -m sbflow.eval_fast --repo-root . \
  --ckpt runs/blog_sbflow/Mid8M_noOT/model_step500.pt \
         runs/blog_sbflow/Mid8M_noOT/model_step1000.pt \
         runs/blog_sbflow/Mid8M_noOT/model_step1500.pt \
         runs/blog_sbflow/Mid8M_noOT/model_step2000.pt \
  --style-a Impressionism --style-b Ukiyo_e \
  --out "$OUT" --steps 4 --method heun --batch 8
echo "=== FAST EVAL DONE ==="
