#!/usr/bin/env bash
# WEAVE R2 P2 Execution Script: Content-Constrained R1 Teacher Construction & Diagnostics.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_r2/queue_p2.log

mkdir -p runs/v6_r2/p2_teachers
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3
qlog "==================== STARTING WEAVE R2 P2 TEACHER EVALUATION ===================="

run_step "p2_teachers_eval" "runs/v6_r2/p2_teachers/p2_teachers_summary.json" \
  python3 -u -m tools.v6_next.run_p2_teachers \
    --out-dir runs/v6_r2/p2_teachers

qlog "==================== P2 TEACHER EVALUATION COMPLETED failures=$QFAIL ===================="
