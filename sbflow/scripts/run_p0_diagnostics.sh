#!/usr/bin/env bash
# WEAVE R2 P0 Execution Script: Unified Baseline Re-evaluations and DINO-S Validity Control.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_r2/queue_p0.log

mkdir -p runs/v6_r2/cache
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3
qlog "==================== STARTING WEAVE R2 P0 ===================="

# 1. Identity (VAE Roundtrip)
run_step "p0_identity" "runs/v6_r2/eval_identity/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen identity --out-dir runs/v6_r2/eval_identity

# 2. Q2a Direct
run_step "p0_Q2a_direct" "runs/v6_r2/eval_Q2a_direct/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    --out-dir runs/v6_r2/eval_Q2a_direct

# 3. Q2a + R1 Oracle (1 step, RMS=0.25)
run_step "p0_Q2a_oracle" "runs/v6_r2/eval_Q2a_oracle/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen oracle --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    --oracle-critic runs/v6_next/probes/r1_r1.pt --oracle-rms 0.25 --oracle-steps 1 \
    --out-dir runs/v6_r2/eval_Q2a_oracle

# 4. B3 distilled lambda=20
run_step "p0_B3_distill20" "runs/v6_r2/eval_B3_distill20/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt runs/v6/P1_B3_distill20/P1_B3_distill20.pt \
    --out-dir runs/v6_r2/eval_B3_distill20

# 5. B4 distilled lambda=50
run_step "p0_B4_distill50" "runs/v6_r2/eval_B4_distill50/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt runs/v6/P1_B4_distill50/P1_B4_distill50.pt \
    --out-dir runs/v6_r2/eval_B4_distill50

# 6. ScoreDiff N=24
run_step "p0_scorediff_N24" "runs/v6_r2/eval_scorediff_N24/eval.json" \
  python3 -u -m tools.v6_next.r2_eval \
    --gen scorediff --sd-ckpt runs/v6/score_diff/score_diff_net.pt \
    --sd-steps 24 --sd-lr 0.100 --sd-anchor 0.08 \
    --out-dir runs/v6_r2/eval_scorediff_N24

# 7. Run DINO-S validity control probe
qlog "Running DINO-S validity control probe..."
python3 -u -m tools.v6_next.dino_s_control > runs/v6_r2/dino_s_control.txt 2>&1
cat runs/v6_r2/dino_s_control.txt

qlog "==================== P0 COMPLETED failures=$QFAIL ===================="
