#!/usr/bin/env bash
# Q2: two SEPARATE mechanisms, each with its own arm so neither can borrow the other's credit.
#
#   Q2a  embedding-space target-style supervision (pull the critic EMBEDDING toward the real
#        target mean) - different objective from ranking, which pushes the LOGIT margin
#   Q2b  content trust region (hinge on latent displacement beyond a radius)
#   Q2c  the two combined, ONLY after both single-mechanism arms exist
#
# Ranking is held at the Q0-winning lambda so the comparison isolates the new term.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source sbflow/scripts/q_lib.sh

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
BASE="--mode direct --base-dim 16 --steps 800 --microbatch 80 --accum 3 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --amp-dtype bf16 --init-from $INIT --rank-critic $CRITIC --rank-tau 0.5"

train () { local name=$1; shift
  python3 -u tools/v6/train_eval_v6.py --name "$name" $BASE "$@"
}

# reuse the winning lambda from Q0 if available, else fall back to 0.001
LAM=$(python3 - <<'PY'
import json
from pathlib import Path
best, bestv, lam = None, -1e9, 0.001
for a in ("G1a_rank_weak", "G1b_rank_strong", "Q1a_rank_00003", "Q1b_rank_0002"):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    v = sum(pt.values()) / len(pt) if pt else 0.0
    if v > bestv:
        best, bestv, lam = a, v, d.get("rank_lambda", 0.001)
print(lam)
PY
)
qlog "Q2: carrying forward lambda_r=$LAM"

run_step "Q2a_embed" "runs/v6/Q2a_embed_style/report.json" \
  train Q2a_embed_style --rank-lambda "$LAM" --embed-style-lambda 0.3
verify_arm Q2a_embed_style || true

run_step "Q2b_trust" "runs/v6/Q2b_trust/report.json" \
  train Q2b_trust --rank-lambda "$LAM" --trust-lambda 0.05 --trust-radius 1.0
verify_arm Q2b_trust || true

run_step "Q2c_both" "runs/v6/Q2c_embed_trust/report.json" \
  train Q2c_embed_trust --rank-lambda "$LAM" --embed-style-lambda 0.3 \
        --trust-lambda 0.05 --trust-radius 1.0
verify_arm Q2c_embed_trust || true

qlog "Q2 done, failures so far: $QFAIL"
