#!/usr/bin/env bash
# P1: can a single-step Direct FIT the teacher edit?
#
# The teacher is z_T = R1Refine(G_Q2a(c, t), t), a deterministic function of (c, t). If Direct cannot
# fit it, the gap is capacity or optimisation; if it fits but the eval does not improve, the gap is
# generalisation or a teacher shortcut.
#
#   B0   218K from Q2a, original loss only        (control for B1)
#   B1   218K from Q2a, + teacher distillation
#   B2c  845K from L8_dim32, original loss only   (control for B2, SAME width, so capacity is
#   B2   845K from L8_dim32, + teacher distillation    attributable independently of distillation)
#
# B2's control is mandatory: without a same-width no-distillation arm, any B2 gain could be width
# rather than distillation.
#
# bf16 throughout: fp16 was measured to underflow small auxiliary gradients to exactly zero.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_p1.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

TEACHER=runs/v6/Q2a_embed_style/Q2a_embed_style.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
Q2A=runs/v6/Q2a_embed_style/Q2a_embed_style.pt
L8=runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt

BASE="--mode direct --steps 800 --microbatch 80 --accum 3 --seed 42 --canon-mode c1_gentle \
--lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 --ref-top-k 1024 --amp-dtype bf16"

DIST="--distill-teacher $TEACHER --distill-critic $CRITIC --distill-steps 1 \
--distill-rms 0.25 --distill-lambda 1.0"

run_arm () {
  local name=$1; shift
  run_step "$name" "runs/v6/${name}/report.json" \
    python3 -u tools/v6/train_eval_v6.py --name "$name" "$@"
  verify_arm "$name" || true
}

# --- 218K: fit or not ---
run_arm P1_B0_ctrl218   $BASE --base-dim 16 --init-from "$Q2A"
run_arm P1_B1_distill218 $BASE --base-dim 16 --init-from "$Q2A" $DIST

# --- 845K: capacity check with its own same-width control ---
run_arm P1_B2c_ctrl845   $BASE --base-dim 32 --init-from "$L8"
run_arm P1_B2_distill845 $BASE --base-dim 32 --init-from "$L8" $DIST

# --- guards: distillation must actually reach the weights, and each pair must differ ---
python3 - <<'PY'
import sys, torch
from pathlib import Path

def m(a):
    p = Path(f"runs/v6/{a}/{a}.pt")
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)["model"]

def d(x, y):
    return max(float((x[k].float() - y[k].float()).abs().max()) for k in x)

bad = 0
for ctrl, dist in (("P1_B0_ctrl218", "P1_B1_distill218"),
                   ("P1_B2c_ctrl845", "P1_B2_distill845")):
    a, b = m(ctrl), m(dist)
    if a is None or b is None:
        print(f"  {ctrl}/{dist}: missing checkpoint"); bad += 1; continue
    v = d(a, b)
    print(f"  {dist} vs {ctrl}: max|dW|={v:.3e} -> "
          + ("FATAL: distillation did nothing" if v == 0 else "distillation reached the weights"))
    if v == 0:
        bad += 1
sys.exit(1 if bad else 0)
PY

qlog "==================== P1 DONE failures=$QFAIL ===================="
