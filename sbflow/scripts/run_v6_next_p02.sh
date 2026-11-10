#!/usr/bin/env bash
# P0.2: reproduce the historical ScoreDiff configs under the SAME scoring protocol as Direct.
#   Cand_A  N=12  lr=0.160  anchor=0.08   (historical Heldout400 Top-1 27.3%, dT +0.0887)
#   Cand_B  N=24  lr=0.100  anchor=0.08   (historical Heldout400 Top-1 32.5%, dT +0.0925)
# Step 1 is a 32-task benchmark to measure NFE cost before committing to all 400.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi

LIMIT=${1:-32}
TAG=${2:-bench}
CKPT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt

echo "########## Cand_A  N=12 lr=0.160 anchor=0.08  (limit=$LIMIT) ##########"
time python3 -u -m tools.v6_24h.eval_tiers --ckpt "$CKPT" --gen scorediff \
  --sd-steps 12 --sd-lr 0.160 --sd-anchor 0.08 \
  --tasks runs/v6_24h/heldout400_tasks.json \
  --out "runs/v6_next/eval_sdA_${TAG}" --limit "$LIMIT"

echo "########## Cand_B  N=24 lr=0.100 anchor=0.08  (limit=$LIMIT) ##########"
time python3 -u -m tools.v6_24h.eval_tiers --ckpt "$CKPT" --gen scorediff \
  --sd-steps 24 --sd-lr 0.100 --sd-anchor 0.08 \
  --tasks runs/v6_24h/heldout400_tasks.json \
  --out "runs/v6_next/eval_sdB_${TAG}" --limit "$LIMIT"
