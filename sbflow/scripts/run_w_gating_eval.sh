#!/usr/bin/env bash
# WEAVE W P2 — evaluate spatially adaptive gated R1 editing on tasks_750.
# Start from the strongest existing baseline (B4 λ=50) and compare gating modes.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_gating_eval.log

mkdir -p runs/v6_w
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE W GATING EVAL ===================="

START_CK=runs/v6/P1_B4_distill50/P1_B4_distill50.pt
COMMON="--gen gated --ckpt $START_CK --gated-critic runs/v6_next/probes/r1_r1.pt --gating-steps 4 --gating-rms 0.08 --gating-topk-frac 0.25"

run_gated () {
  local mode=$1
  local out="runs/v6_w/eval_gated_${mode}/eval.json"
  if [ -s "$out" ]; then qlog "===== eval_gated_${mode} ALREADY DONE ====="; return 0; fi
  run_step "eval_gated_${mode}" "$out" \
    python3 -u -m tools.v6_next.r2_eval --tasks eval_protocol/tasks_750.json \
      --out-dir "runs/v6_w/eval_gated_${mode}" $COMMON --gating-mode "$mode"
}

MODES="${GATING_MODES:-full hf topk content style_energy}"
for mode in $MODES; do
  run_gated "$mode"
done

qlog "==================== GATING EVAL DONE failures=$QFAIL ===================="
