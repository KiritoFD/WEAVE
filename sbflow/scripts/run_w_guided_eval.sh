#!/usr/bin/env bash
# WEAVE W P1 — evaluate guided latent editing on the unified 750-task protocol.
# Starts from the best W student and compares:
#   guided_r1     : multi-step R1 margin gradient only
#   guided_score  : real-latent score prior only
#   guided_both   : normalized R1 + real-latent score
# All use the same per-step RMS budget and number of steps.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_guided_eval.log

mkdir -p runs/v6_w
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE W GUIDED EDIT EVAL ===================="

BEST=$(python3 -m tools.v6_next.w_select_best)
qlog "Best W student: $BEST"

GUIDED_ARGS="--gen guided --ckpt $BEST --guided-critic runs/v6_next/probes/r1_r1.pt --guided-score-ckpt runs/v6_w/score_prior/score_prior.pt --guided-steps 4 --guided-rms 0.08 --guided-sigma 0.30 --guided-r1-weight 1.0 --guided-score-weight 1.0"

run_guided () {
  local name=$1 mode=$2
  local out="runs/v6_w/eval_guided_${name}/eval.json"
  if [ -s "$out" ]; then qlog "===== eval_guided_${name} ALREADY DONE ====="; return 0; fi
  run_step "eval_guided_${name}" "$out" \
    python3 -u -m tools.v6_next.r2_eval --tasks eval_protocol/tasks_750.json \
      --out-dir "runs/v6_w/eval_guided_${name}" $GUIDED_ARGS --guided-mode "$mode"
}

run_guided r1 r1
run_guided score score
run_guided both both

qlog "==================== GUIDED EDIT EVAL DONE failures=$QFAIL ===================="
