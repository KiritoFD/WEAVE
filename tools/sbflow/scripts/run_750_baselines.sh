#!/usr/bin/env bash
# Complete the baseline for the formal ranking set: L11 (the original lightweight Direct) had never
# been evaluated on tasks_750, only on Heldout400. Without it there is no "original baseline" row.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_750_baselines.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

for arm in L11_k1024_dim16 Q2a_embed_style; do
  ck="runs/v6/${arm}/${arm}.pt"
  [ -s "$ck" ] || { qlog "missing $ck"; QFAIL=$((QFAIL+1)); continue; }
  run_step "750_${arm}" "runs/v6_next/eval_750_${arm}/eval.json" \
    python3 -u -m tools.v6_24h.eval_tiers --ckpt "$ck" \
      --tasks eval_protocol/tasks_750.json --out "runs/v6_next/eval_750_${arm}"
done

qlog "==================== BASELINES DONE failures=$QFAIL ===================="
