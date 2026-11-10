#!/usr/bin/env bash
# Q1: replicate the G1b-vs-G0 difference at a second seed.
#
# The lambda sweep is SKIPPED deliberately. Both lambdas (0.001 and 0.003) gave exactly the same
# overall Top-1 (+1.0pp) and the same per-target pattern (only Rococo moved), so a third lambda would
# add little. The open question is whether the Rococo gain is real or four lucky tasks out of 80, and
# that is answered by replication, not by more lambdas.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_q1.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
COMMON="--mode direct --base-dim 16 --steps 800 --microbatch 80 --accum 3 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --amp-dtype bf16 --init-from $INIT"

run_arm () {
  local name=$1; shift
  run_step "$name" "runs/v6/${name}/report.json" \
    python3 -u tools/v6/train_eval_v6.py --name "$name" $COMMON "$@"
  verify_arm "$name" || true
}

run_arm Q1_G0_s43
run_arm Q1_G1b_s43 --rank-critic "$CRITIC" --rank-lambda 0.003 --rank-tau 0.5

# the guard: the two seed-43 arms must differ
python3 - <<'PY'
import sys, torch
base = torch.load("runs/v6/Q1_G0_s43/Q1_G0_s43.pt", map_location="cpu",
                  weights_only=False)["model"]
b = torch.load("runs/v6/Q1_G1b_s43/Q1_G1b_s43.pt", map_location="cpu",
               weights_only=False)["model"]
mx = max(float((base[k].float()-b[k].float()).abs().max()) for k in base)
print(f"\nQ1 seed43: max|dW|={mx:.3e} -> "
      + ("DID NOTHING" if mx == 0.0 else "ranking term reached the weights"))
sys.exit(1 if mx == 0.0 else 0)
PY

qlog "==================== Q1 DONE failures=$QFAIL ===================="
