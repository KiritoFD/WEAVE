#!/usr/bin/env bash
# Q1: ranking-weight sweep + second-seed paired reproduction.
#
# Q0 decides which G1 arm to carry forward; that arm is reproduced at seed 43 so the effect can be
# separated from training randomness. The sweep brackets the current operating point
# (lambda_r 0.001) with 0.0003 and 0.002, i.e. roughly gradient ratios 0.03 and 0.22.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source sbflow/scripts/q_lib.sh

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
BASE="--mode direct --base-dim 16 --steps 800 --microbatch 80 --accum 3 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --amp-dtype bf16 --init-from $INIT"

train () {  # train <name> <seed> <extra...>
  local name=$1; local seed=$2; shift 2
  python3 -u tools/v6/train_eval_v6.py --name "$name" $BASE --seed "$seed" "$@"
}

# --- sweep around the current operating point ---
run_step "Q1a_lr00003" "runs/v6/Q1a_rank_00003/report.json" \
  train Q1a_rank_00003 42 --rank-critic "$CRITIC" --rank-lambda 0.0003 --rank-tau 0.5
verify_arm Q1a_rank_00003 || true

run_step "Q1b_lr002" "runs/v6/Q1b_rank_0002/report.json" \
  train Q1b_rank_0002 42 --rank-critic "$CRITIC" --rank-lambda 0.002 --rank-tau 0.5
verify_arm Q1b_rank_0002 || true

# --- second seed on the G0 control and on the strongest G1 arm (chosen by Q0) ---
BEST=$(python3 - <<'PY'
import json
from pathlib import Path
best, bestv = "G1a_rank_weak", -1e9
for a in ("G1a_rank_weak", "G1b_rank_strong"):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        continue
    e = json.loads(p.read_text()).get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    v = sum(pt.values()) / len(pt) if pt else 0.0
    if v > bestv:
        best, bestv = a, v
print(best)
PY
)
qlog "Q1: best G1 arm from Q0 = $BEST"

# read the winning arm's lambda back from its report so the seed-43 run matches it exactly
LAM=$(python3 - "$BEST" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(f"runs/v6/{sys.argv[1]}/report.json").read_text())
print(d.get("rank_lambda", 0.001))
PY
)
qlog "Q1: reproducing $BEST at seed 43 with lambda_r=$LAM"

run_step "Q1c_G0_seed43" "runs/v6/Q1c_G0_s43/report.json" \
  train Q1c_G0_s43 43
verify_arm Q1c_G0_s43 || true

run_step "Q1d_best_seed43" "runs/v6/Q1d_G1_s43/report.json" \
  train Q1d_G1_s43 43 --rank-critic "$CRITIC" --rank-lambda "$LAM" --rank-tau 0.5
verify_arm Q1d_G1_s43 || true

qlog "Q1 done, failures so far: $QFAIL"
