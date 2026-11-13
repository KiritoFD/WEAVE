#!/usr/bin/env bash
# Re-run student fitting diagnostics (fix: create per-arm out dir), then P4 final eval.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_r2/queue_p3diag.log

source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== P3 FITTING DIAGNOSTICS (retry) ===================="

mkdir -p runs/v6_r2/fitting_diagnostics
for s_arm in S1_distill20_proj_ll S2_distill50_proj_ll S3_ctrl845_distill20_oracle S4_ctrl845_distill20_proj_ll; do
  t_mode="projected_ll"
  if [[ "$s_arm" == *"oracle"* ]]; then t_mode="oracle"; fi
  mkdir -p "runs/v6_r2/fitting_diagnostics/${s_arm}"
  qlog "--- fitting diag: $s_arm (teacher=$t_mode) ---"
  python3 -u -m tools.v6_next.student_fit_diag \
    --student-ckpt "runs/v6_r2/${s_arm}/${s_arm}.pt" \
    --teacher-mode "$t_mode" \
    --out-dir "runs/v6_r2/fitting_diagnostics/${s_arm}" 2>&1 | tee -a "$QLOG"
done

qlog "==================== P3 DIAGNOSTICS COMPLETED ===================="
