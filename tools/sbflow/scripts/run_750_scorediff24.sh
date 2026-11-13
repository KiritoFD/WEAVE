#!/usr/bin/env bash
# ScoreDiff N=24 through eval_tiers so it shares a reference convention with the checkpoint rows.
# The P0 run measured it against the VAE round-trip source, which is why its DINO-C/LPIPS could not
# be placed beside the eval_tiers rows. --gen scorediff already exists; no code change is needed.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_750_sd24.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

run_step "750_scorediff_N24" "runs/v6_next/eval_750_scorediff_N24/eval.json" \
  python3 -u -m tools.v6_24h.eval_tiers \
    --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    --gen scorediff --sd-steps 24 --sd-lr 0.100 --sd-anchor 0.08 \
    --tasks eval_protocol/tasks_750.json --out runs/v6_next/eval_750_scorediff_N24

qlog "==================== SD24 DONE failures=$QFAIL ===================="
