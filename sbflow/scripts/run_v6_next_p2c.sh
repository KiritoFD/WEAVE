#!/usr/bin/env bash
# P2 with bf16 AMP: G0 (control) + G1a (weak) + G1b (strong).
#
# WHY bf16 AND NOT fp16: fp16 underflows the ranking gradient to EXACTLY zero. Measured on 4-step
# twins at the real microbatch/accum, comparing weights against a no-rank run:
#
#   fp16   lambda 0.003 -> max|dW| = 0.000e+00   0/113 tensors
#   fp16   lambda 0.010 -> max|dW| = 0.000e+00   0/113
#   fp16   lambda 0.020 -> max|dW| = 1.316e-03  113/113
#   fp16   lambda 0.050 -> max|dW| = 1.527e-03  113/113
#   bf16   lambda 0.001 -> max|dW| = 1.465e-03  113/113
#   bf16   lambda 0.003 -> max|dW| = 1.509e-03  113/113
#
# So under fp16 the only usable lambdas are >= 0.02, i.e. gradient ratios >= 2.2, which would let
# the ranking term dominate. bf16 has fp32's exponent range and removes the cliff entirely.
#
# ALL THREE arms are re-run on bf16 because the AMP dtype changes numerics, so a bf16 G1 compared
# against an fp16 G0 would not be a controlled comparison.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_p2c.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
COMMON="--mode direct --base-dim 16 --steps 800 --microbatch 80 --accum 3 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --amp-dtype bf16 --init-from $INIT"

run_arm () {
  local name=$1; shift
  run_step "$name" "runs/v6/${name}/report.json" \
    python3 -u tools/v6/train_eval_v6.py --name "$name" $COMMON "$@"
  verify_arm "$name" || true
}

run_arm P2_G0_bf16
run_arm P2_G1a_bf16 --rank-critic "$CRITIC" --rank-lambda 0.001 --rank-tau 0.5
run_arm P2_G1b_bf16 --rank-critic "$CRITIC" --rank-lambda 0.003 --rank-tau 0.5

# hard guard: the ranking arms must NOT be bit-identical to the control
python3 - <<'PY'
import sys, torch
from pathlib import Path
base = torch.load("runs/v6/P2_G0_bf16/P2_G0_bf16.pt", map_location="cpu",
                  weights_only=False)["model"]
bad = 0
for a in ("P2_G1a_bf16", "P2_G1b_bf16"):
    p = Path(f"runs/v6/{a}/{a}.pt")
    if not p.exists():
        print(f"  {a}: missing"); bad += 1; continue
    b = torch.load(p, map_location="cpu", weights_only=False)["model"]
    mx = max(float((base[k].float()-b[k].float()).abs().max()) for k in base)
    nd = sum(1 for k in base if float((base[k].float()-b[k].float()).abs().max()) > 0)
    print(f"  {a} vs P2_G0_bf16: max|dW|={mx:.3e} tensors_differing={nd}/{len(base)}")
    if mx == 0.0:
        print("    FATAL: identical to the control -- the ranking term did not reach the weights")
        bad += 1
sys.exit(1 if bad else 0)
PY

qlog "==================== P2c DONE failures=$QFAIL ===================="
