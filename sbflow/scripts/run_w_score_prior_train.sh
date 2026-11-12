#!/usr/bin/env bash
# WEAVE W P1 — train the unconditional real-latent score prior on A0 latents.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_score_prior.log

mkdir -p runs/v6_w
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE W SCORE PRIOR TRAIN ===================="

run_step "train_score_prior" "runs/v6_w/score_prior/score_prior.pt" \
  python3 -u -m tools.v6_next.w_score_prior \
    --latent-dir pseudo_gt_24h/orig_mean_fp16_a0 \
    --out-dir runs/v6_w/score_prior \
    --base-dim 64 --steps 1000 --microbatch 128 --lr 0.0003 --seed 42

qlog "==================== SCORE PRIOR TRAIN DONE failures=$QFAIL ===================="
