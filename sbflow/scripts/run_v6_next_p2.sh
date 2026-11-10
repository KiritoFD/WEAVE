#!/usr/bin/env bash
# P2: G0 (control) + G1a (weak ranking) + G1b (strong ranking).
#
# All three start from the SAME L11 checkpoint with the same seed, data order, optimizer init and LR
# schedule, so the only difference is the ranking loss. Optimizer state is not restored because the
# original run never saved it; every arm restarts it identically from step 0.
#
# lambda_r is CALIBRATED, not guessed. A 2-step diagnostic at lambda_r=0.1 measured
#   |grad(lam_r * L_rank)| = 1.682e+01  vs  |grad(L_V6)| = 1.513e+00
# i.e. a gradient ratio of 11.12, so the ranking term would dominate the V6 objective by 11x.
# The ratio scales linearly in lambda_r, so ratio/lambda_r = 111.2 and:
#   target ratio 0.10  ->  lambda_r = 0.10 / 111.2 = 0.0009
#   target ratio 0.30  ->  lambda_r = 0.30 / 111.2 = 0.0027
# We use 0.001 and 0.003, and the step-1 print in each arm verifies the achieved ratio.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
COMMON="--mode direct --base-dim 16 --steps 800 --microbatch 80 --accum 3 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --init-from $INIT"

run_arm () {
  local name=$1; shift
  echo
  echo "##################### $name #####################"
  python3 -u tools/v6/train_eval_v6.py --name "$name" $COMMON "$@" 2>&1 | tee -a "runs/v6_24h/p2_${name}.log"
}

run_arm G0_ctrl

run_arm G1a_rank_weak   --rank-critic "$CRITIC" --rank-lambda 0.001 --rank-tau 0.5

run_arm G1b_rank_strong --rank-critic "$CRITIC" --rank-lambda 0.003 --rank-tau 0.5

echo
echo "##################### P2 TRAINING DONE #####################"
