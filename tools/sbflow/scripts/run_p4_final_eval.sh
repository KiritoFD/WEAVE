#!/usr/bin/env bash
# WEAVE R2 P4 Execution Script: 750 Tasks Evaluation, Paired Analysis, Visual Grids.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_r2/queue_p4.log

mkdir -p runs/v6_r2
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3
qlog "==================== STARTING WEAVE R2 P4 FINAL EVAL ===================="

# 1. Evaluate S1 (218K, P2 Projected-LL Teacher, lambda=20)
run_step "eval_S1" "runs/v6_r2/eval_S1/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt runs/v6_r2/S1_distill20_proj_ll/S1_distill20_proj_ll.pt \
    --out-dir runs/v6_r2/eval_S1

# 2. Evaluate S2 (218K, P2 Projected-LL Teacher, lambda=50)
run_step "eval_S2" "runs/v6_r2/eval_S2/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt runs/v6_r2/S2_distill50_proj_ll/S2_distill50_proj_ll.pt \
    --out-dir runs/v6_r2/eval_S2

# 3. Evaluate S3 (845K, Oracle Teacher, lambda=20)
run_step "eval_S3" "runs/v6_r2/eval_S3/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt runs/v6_r2/S3_ctrl845_distill20_oracle/S3_ctrl845_distill20_oracle.pt \
    --out-dir runs/v6_r2/eval_S3

# 4. Evaluate S4 (845K, P2 Projected-LL Teacher, lambda=20)
run_step "eval_S4" "runs/v6_r2/eval_S4/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt runs/v6_r2/S4_ctrl845_distill20_proj_ll/S4_ctrl845_distill20_proj_ll.pt \
    --out-dir runs/v6_r2/eval_S4

# 5. Evaluate B2c (845K control, no distillation)
run_step "eval_P1_B2c_ctrl845" "runs/v6_r2/eval_P1_B2c_ctrl845/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt runs/v6/P1_B2c_ctrl845/P1_B2c_ctrl845.pt \
    --out-dir runs/v6_r2/eval_P1_B2c_ctrl845

# 6. Run Paired Cluster Bootstrap Comparison across ALL models
qlog "Running Paired Cluster Bootstrap Comparison..."
python3 -u -m tools.v6_next.paired_comparison_r2 > runs/v6_r2/final_paired_report.txt 2>&1
cat runs/v6_r2/final_paired_report.txt

# 7. Generate Visual Grids
qlog "Generating visual comparison grids..."
python3 -u -m tools.v6_next.make_visual_grids \
  --new-ckpt runs/v6_r2/S1_distill20_proj_ll/S1_distill20_proj_ll.pt \
  --out-dir runs/v6_r2/visual_grids

qlog "==================== P4 FINAL EVAL COMPLETED failures=$QFAIL ===================="
