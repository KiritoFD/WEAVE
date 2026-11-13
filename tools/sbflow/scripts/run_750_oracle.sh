#!/usr/bin/env bash
# Score the Oracle through the SAME eval_tiers path as every checkpoint.
#
# P0 measured the Oracle with amortize_p0.py, whose DINO-C and LPIPS are computed against the VAE
# ROUND-TRIP source rather than the original PNG. That makes its content numbers systematically
# lower than eval_tiers' (LPIPS 0.273 vs 0.400 for the identical Q2a Direct), so the Oracle could not
# be placed on the same ranking as the checkpoints. This run fixes that.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_750_oracle.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

for spec in "1:0.25:oracle_s1" "4:0.25:oracle_s4"; do
  steps=${spec%%:*}
  rest=${spec#*:}
  rms=${rest%%:*}
  tag=${rest##*:}
  run_step "750_${tag}" "runs/v6_next/eval_750_${tag}/eval.json" \
    python3 -u -m tools.v6_24h.eval_tiers \
      --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
      --gen oracle --oracle-steps "$steps" --oracle-rms "$rms" \
      --tasks eval_protocol/tasks_750.json --out "runs/v6_next/eval_750_${tag}"
done

qlog "==================== ORACLE DONE failures=$QFAIL ===================="
