#!/usr/bin/env bash
# P2: G0 (control) + G1a (weak ranking) + G1b (strong ranking).
#
# FIXED after review: the original version used `set -uo pipefail` with `... | tee`, so a training
# failure was masked (the pipeline status is tee's) and nothing aborted. Every arm now goes through
# q_lib's run_step, which reads PIPESTATUS[0] before tee can overwrite it and then verifies the
# checkpoint and report, so a printed "DONE" can never be read as success.
#
# lambda_r is CALIBRATED, not guessed. A 2-step diagnostic at lambda_r=0.1 measured
#   |grad(lam_r * L_rank)| = 1.682e+01  vs  |grad(L_V6)| = 1.513e+00
# i.e. a gradient ratio of 11.12, so a fixed coefficient would have let the ranking term swamp the
# V6 objective by 11x. The ratio scales linearly, so ratio/lambda_r = 111.2 and:
#   target ratio 0.10  ->  lambda_r = 0.0009
#   target ratio 0.30  ->  lambda_r = 0.0027
# The arms use 0.001 and 0.003.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_p2.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
COMMON="--mode direct --base-dim 16 --steps 800 --microbatch 80 --accum 3 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --init-from $INIT"

run_arm () {
  local name=$1; shift
  run_step "$name" "runs/v6/${name}/report.json" \
    python3 -u tools/v6/train_eval_v6.py --name "$name" $COMMON "$@"
  verify_arm "$name" || true
}

run_arm G0_ctrl
run_arm G1a_rank_weak   --rank-critic "$CRITIC" --rank-lambda 0.001 --rank-tau 0.5
run_arm G1b_rank_strong --rank-critic "$CRITIC" --rank-lambda 0.003 --rank-tau 0.5

qlog "==================== P2 TRAINING DONE failures=$QFAIL ===================="
[ "$QFAIL" -eq 0 ] || exit 1
