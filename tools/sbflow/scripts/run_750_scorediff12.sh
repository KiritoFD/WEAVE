#!/usr/bin/env bash
# Fill the one gap in the delivery table that this session can actually measure: ScoreDiff at N=12
# with the strong-style step (lr 0.160) on the full 750 protocol. The N=24 row is already measured.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_750_sd12.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

run_step "750_scorediff_N12" "runs/v6_next/eval_750_scorediff_N12/eval.json" \
  python3 -u -m tools.v6_24h.eval_tiers \
    --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    --gen scorediff --sd-steps 12 --sd-lr 0.160 --sd-anchor 0.08 \
    --tasks eval_protocol/tasks_750.json --out runs/v6_next/eval_750_scorediff_N12

qlog "==================== SD12 DONE failures=$QFAIL ===================="
