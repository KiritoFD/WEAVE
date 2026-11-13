#!/usr/bin/env bash
# The lambda=1.0 distillation weight was ~20x too small. At 200 steps lambda=20 already reaches
# Top-1 0.145 against the control's 0.060, and Ukiyo moves off zero for the first time. Run the full
# 800 steps at lambda 20 and 50 to find the operating point.
#
# lambda is CALIBRATED here, not guessed: the fit diagnostic shows rel_move scaling roughly linearly
# in lambda (0.0032 / 0.0121 / 0.0451 at lambda 1 / 5 / 20), so the teacher's edit is representable
# and the question is only how much of the V6 objective to trade for it.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_p1b.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

Q2A=runs/v6/Q2a_embed_style/Q2a_embed_style.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
C="--mode direct --base-dim 16 --steps 800 --microbatch 80 --accum 3 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --amp-dtype bf16 --init-from $Q2A"

run_arm () {
  local name=$1; shift
  run_step "$name" "runs/v6/${name}/report.json" \
    python3 -u tools/v6/train_eval_v6.py --name "$name" "$@"
  verify_arm "$name" || true
}

run_arm P1_B3_distill20 $C --distill-teacher "$Q2A" --distill-critic "$CRITIC" \
  --distill-steps 1 --distill-rms 0.25 --distill-lambda 20
run_arm P1_B4_distill50 $C --distill-teacher "$Q2A" --distill-critic "$CRITIC" \
  --distill-steps 1 --distill-rms 0.25 --distill-lambda 50

python3 - <<'PY'
import sys, torch
from pathlib import Path
def m(a):
    p = Path(f"runs/v6/{a}/{a}.pt")
    return torch.load(p, map_location="cpu", weights_only=False)["model"] if p.exists() else None
def d(x, y):
    return max(float((x[k].float()-y[k].float()).abs().max()) for k in x)
base = m("P1_B0_ctrl218")
bad = 0
for a in ("P1_B3_distill20", "P1_B4_distill50"):
    b = m(a)
    if b is None or base is None:
        print(f"  {a}: missing"); bad += 1; continue
    v = d(base, b)
    print(f"  {a} vs B0_ctrl218: max|dW|={v:.3e} -> "
          + ("FATAL: distill did nothing" if v == 0 else "reached the weights"))
    bad += int(v == 0)
sys.exit(1 if bad else 0)
PY

qlog "==================== P1b DONE failures=$QFAIL ===================="
