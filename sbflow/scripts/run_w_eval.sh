#!/usr/bin/env bash
# WEAVE W P4 — final 750-task evaluation of all W arms (unified r2_eval protocol).
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_eval.log

mkdir -p runs/v6_w
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE W FINAL 750 EVAL ===================="

eval_arm () {
  local name=$1
  local ck="runs/v6_w/${name}/${name}.pt"
  local out="runs/v6_w/eval_${name}/eval.json"
  [ -s "$ck" ] || { qlog "SKIP $name (no checkpoint)"; return 0; }
  run_step "eval_$name" "$out" \
    python3 -u -m tools.v6_next.r2_eval \
      --gen direct --ckpt "$ck" --out-dir "runs/v6_w/eval_${name}"
}

for arm in W1_FeatDistill_218 W2_LossClean_218 W3_LossClean_218_l50 \
           W4_LossClean_845_SK; do
  eval_arm "$arm"
done

qlog "==================== W FINAL EVAL DONE failures=$QFAIL ===================="
