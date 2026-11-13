#!/usr/bin/env bash
# WEAVE R2 P1 Execution Script: R1 Gradient Mechanism & Subband Scan.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_r2/queue_p1.log

mkdir -p runs/v6_r2/p1_analysis
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3
qlog "==================== STARTING WEAVE R2 P1 ANALYSIS ===================="

run_step "p1_gradient_scan" "runs/v6_r2/p1_analysis/p1_summary.json" \
  python3 -u -m tools.v6_next.run_p1_analysis \
    --out-dir runs/v6_r2/p1_analysis

qlog "==================== P1 ANALYSIS COMPLETED failures=$QFAIL ===================="
