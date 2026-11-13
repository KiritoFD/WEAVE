#!/usr/bin/env bash
# Q1: replicate the G1b-vs-G0 difference at a SECOND SEED.
#
# NOTE ON A BUG THIS SCRIPT ALREADY HAD: the first version never passed --seed 43, so both "seed-43"
# arms were silently seed 42 duplicates (report.json showed seed=42 and the metrics matched
# P2_G0_bf16 to six decimals). The queue now asserts the seed that actually landed in the report, so
# a forgotten flag fails loudly instead of producing a fake replication.
#
# The lambda sweep is skipped deliberately: lambda 0.001 and 0.003 gave identical overall Top-1 and
# an identical per-target pattern, so a third lambda carries no information. What matters is whether
# the Rococo gain replicates.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_q1b.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

SEED=43
INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
COMMON="--mode direct --base-dim 16 --steps 800 --microbatch 80 --accum 3 --seed $SEED \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --amp-dtype bf16 --init-from $INIT"

check_seed () {
  local name=$1
  local got
  got=$(python3 -c "import json;print(json.load(open('runs/v6/$name/report.json'))['seed'])")
  if [ "$got" != "$SEED" ]; then
    qlog "FATAL: $name ran with seed=$got, expected $SEED -- replication invalid"
    return 1
  fi
  qlog "  seed verified: $name seed=$got"
}

run_arm () {
  local name=$1; shift
  run_step "$name" "runs/v6/${name}/report.json" \
    python3 -u tools/v6/train_eval_v6.py --name "$name" $COMMON "$@"
  verify_arm "$name" || true
  check_seed "$name" || QFAIL=$((QFAIL+1))
}

run_arm Q1b_G0_s43
run_arm Q1b_G1b_s43 --rank-critic "$CRITIC" --rank-lambda 0.003 --rank-tau 0.5

# the arms must differ, and they must NOT equal the seed-42 twins
python3 - <<'PY'
import sys, torch
from pathlib import Path
def m(a):
    return torch.load(f"runs/v6/{a}/{a}.pt", map_location="cpu", weights_only=False)["model"]
def d(x, y):
    return max(float((x[k].float()-y[k].float()).abs().max()) for k in x)
a, b = m("Q1b_G0_s43"), m("Q1b_G1b_s43")
print(f"\n  seed43 G1b vs G0: max|dW|={d(a,b):.3e} -> "
      + ("FATAL: did nothing" if d(a,b) == 0 else "ranking reached the weights"))
s42 = m("P2_G0_bf16")
print(f"  seed43 G0 vs seed42 G0: max|dW|={d(a,s42):.3e} -> "
      + ("FATAL: the seed did not change anything" if d(a,s42) == 0
         else "the seed genuinely differs"))
sys.exit(1 if (d(a,b) == 0 or d(a,s42) == 0) else 0)
PY

qlog "==================== Q1b DONE failures=$QFAIL ===================="
