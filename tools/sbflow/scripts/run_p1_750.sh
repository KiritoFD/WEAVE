#!/usr/bin/env bash
# The plan requires every formal result to run the full cross600/same150 protocol. Val200 selection is
# done; this is the formal evaluation.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_p1_750.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

for arm in P1_B3_distill20 P1_B4_distill50 P1_B0_ctrl218; do
  ck="runs/v6/${arm}/${arm}.pt"
  [ -s "$ck" ] || { qlog "missing $ck"; QFAIL=$((QFAIL+1)); continue; }
  run_step "750_${arm}" "runs/v6_next/eval_750_${arm}/eval.json" \
    python3 -u -m tools.v6_24h.eval_tiers --ckpt "$ck" \
      --tasks eval_protocol/tasks_750.json --out "runs/v6_next/eval_750_${arm}"
done

qlog "==================== 750 EVAL DONE failures=$QFAIL ===================="
