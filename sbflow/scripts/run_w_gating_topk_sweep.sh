#!/usr/bin/env bash
# WEAVE W P2b — focused TopK gating sweep on tasks_750.
# Keeps the same total RMS budget for the step-count variant:
#   4 steps x 0.08 = 8 steps x 0.04.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_gating_topk_sweep.log

mkdir -p runs/v6_w
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE W TOPK GATING SWEEP ===================="

START_CK=runs/v6/P1_B4_distill50/P1_B4_distill50.pt
COMMON="--gen gated --ckpt $START_CK --gated-critic runs/v6_next/probes/r1_r1.pt --gating-mode topk"

run_topk () {
  local name=$1 frac=$2 steps=$3 rms=$4
  local out="runs/v6_w/eval_gated_${name}/eval.json"
  if [ -s "$out" ]; then qlog "===== eval_gated_${name} ALREADY DONE ====="; return 0; fi
  run_step "eval_gated_${name}" "$out" \
    python3 -u -m tools.v6_next.r2_eval --tasks eval_protocol/tasks_750.json \
      --out-dir "runs/v6_w/eval_gated_${name}" $COMMON \
      --gating-topk-frac "$frac" --gating-steps "$steps" --gating-rms "$rms"
}

run_topk topk10 0.10 4 0.08
run_topk topk50 0.50 4 0.08
run_topk topk25_s8 0.25 8 0.04

qlog "==================== TOPK GATING SWEEP DONE failures=$QFAIL ===================="
