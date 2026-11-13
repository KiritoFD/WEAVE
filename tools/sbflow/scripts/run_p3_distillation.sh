#!/usr/bin/env bash
# WEAVE R2 P3 Execution Script: Student Distillation Matrix (S1, S2, S3, S4) & Fitting Diagnostics.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_r2/queue_p3.log

mkdir -p runs/v6_r2
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3
qlog "==================== STARTING WEAVE R2 P3 DISTILLATION ===================="

# Common training hyperparams
COMMON_ARGS="--mode direct --out-base runs/v6_r2 --steps 800 --lr 0.0003 --microbatch 80 --accum 3 --amp-dtype bf16 --canon-mode c1_gentle --w-rec 1.0 --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 --ref-top-k 1024 --seed 42"

# 1. S1: 218K + P2 Projected-LL Teacher, lambda=20
if [ -s "runs/v6_r2/S1_distill20_proj_ll/S1_distill20_proj_ll.pt" ]; then
  qlog "===== STEP train_S1_distill20_proj_ll (ALREADY TRAINED, REUSING) ====="
else
  run_step "train_S1_distill20_proj_ll" "runs/v6_r2/S1_distill20_proj_ll/S1_distill20_proj_ll.pt" \
    python3 -u -m tools.v6.train_eval_v6 \
      --name S1_distill20_proj_ll \
      --base-dim 16 \
      --init-from runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
      --distill-teacher runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
      --distill-critic runs/v6_next/probes/r1_r1.pt \
      --distill-mode projected_ll \
      --distill-steps 1 --distill-rms 0.25 --distill-lambda 20.0 \
      $COMMON_ARGS
fi

# 2. S2: 218K + P2 Projected-LL Teacher, lambda=50
run_step "train_S2_distill50_proj_ll" "runs/v6_r2/S2_distill50_proj_ll/S2_distill50_proj_ll.pt" \
  python3 -u -m tools.v6.train_eval_v6 \
    --name S2_distill50_proj_ll \
    --base-dim 16 \
    --init-from runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    --distill-teacher runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    --distill-critic runs/v6_next/probes/r1_r1.pt \
    --distill-mode projected_ll \
    --distill-steps 1 --distill-rms 0.25 --distill-lambda 50.0 \
    $COMMON_ARGS

# 3. S3: 845K + Original Oracle, lambda=20
run_step "train_S3_ctrl845_distill20_oracle" "runs/v6_r2/S3_ctrl845_distill20_oracle/S3_ctrl845_distill20_oracle.pt" \
  python3 -u -m tools.v6.train_eval_v6 \
    --name S3_ctrl845_distill20_oracle \
    --base-dim 32 \
    --init-from runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt \
    --distill-teacher runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    --distill-critic runs/v6_next/probes/r1_r1.pt \
    --distill-mode oracle \
    --distill-steps 1 --distill-rms 0.25 --distill-lambda 20.0 \
    $COMMON_ARGS

# 4. S4: 845K + P2 Projected-LL Teacher, lambda=20
run_step "train_S4_ctrl845_distill20_proj_ll" "runs/v6_r2/S4_ctrl845_distill20_proj_ll/S4_ctrl845_distill20_proj_ll.pt" \
  python3 -u -m tools.v6.train_eval_v6 \
    --name S4_ctrl845_distill20_proj_ll \
    --base-dim 32 \
    --init-from runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt \
    --distill-teacher runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    --distill-critic runs/v6_next/probes/r1_r1.pt \
    --distill-mode projected_ll \
    --distill-steps 1 --distill-rms 0.25 --distill-lambda 20.0 \
    $COMMON_ARGS

# 5. Run Student Fitting Diagnostics
qlog "Running student fitting diagnostics..."
mkdir -p runs/v6_r2/fitting_diagnostics
for s_arm in S1_distill20_proj_ll S2_distill50_proj_ll S3_ctrl845_distill20_oracle S4_ctrl845_distill20_proj_ll; do
  t_mode="projected_ll"
  if [[ "$s_arm" == *"oracle"* ]]; then t_mode="oracle"; fi
  python3 -u -m tools.v6_next.student_fit_diag \
    --student-ckpt "runs/v6_r2/${s_arm}/${s_arm}.pt" \
    --teacher-mode "$t_mode" \
    --out-dir "runs/v6_r2/fitting_diagnostics/${s_arm}"
done

qlog "==================== P3 DISTILLATION COMPLETED failures=$QFAIL ===================="
