#!/usr/bin/env bash
# WEAVE VAE-swap — 750-task evaluation of the A0-pool arms.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_vaeswap_eval.log

mkdir -p runs/v6_w
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE VAE-SWAP FINAL 750 EVAL ===================="

for arm in V1_A0_LossClean V2_A0_PixelMSE; do
  ck="runs/v6_w/${arm}/${arm}.pt"
  out="runs/v6_w/eval_${arm}/eval.json"
  [ -s "$ck" ] || { qlog "SKIP $arm (no checkpoint)"; continue; }
  run_step "eval_$arm" "$out" \
    python3 -u -m tools.v6_next.r2_eval \
      --gen direct --ckpt "$ck" --out-dir "runs/v6_w/eval_${arm}"
done

qlog "==================== VAE-SWAP EVAL DONE failures=$QFAIL ===================="
