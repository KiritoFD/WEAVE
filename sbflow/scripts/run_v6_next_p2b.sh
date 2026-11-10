#!/usr/bin/env bash
# Re-run only G1a and G1b. G0_ctrl already completed and is valid; re-running it would waste 12
# minutes of GPU for no information.
#
# Context: the first G1a/G1b attempt produced weights BIT-IDENTICAL to G0 because the step-1
# gradient probes (autograd.grad with retain_graph=True on the training graph) silently left the
# ranking branch contributing zero to the backward. Probes are now removed from the training path
# and a probe-free 3-step control confirms the term changes 113/113 tensors.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_p2b.log
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

run_arm G1a_rank_weak   --rank-critic "$CRITIC" --rank-lambda 0.001 --rank-tau 0.5
run_arm G1b_rank_strong --rank-critic "$CRITIC" --rank-lambda 0.003 --rank-tau 0.5

# hard check: G1a must NOT be bit-identical to G0, otherwise the term did nothing again
python3 - <<'PY'
import sys, torch
from pathlib import Path
p0, p1 = Path("runs/v6/G0_ctrl/G0_ctrl.pt"), Path("runs/v6/G1a_rank_weak/G1a_rank_weak.pt")
if not (p0.exists() and p1.exists()):
    print("  cannot compare: missing checkpoint"); sys.exit(1)
a = torch.load(p0, map_location="cpu", weights_only=False)["model"]
b = torch.load(p1, map_location="cpu", weights_only=False)["model"]
mx = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
nd = sum(1 for k in a if float((a[k].float() - b[k].float()).abs().max()) > 0)
print(f"\nG1a vs G0: max|dW|={mx:.3e} tensors_differing={nd}/{len(a)}")
if mx == 0.0:
    print("  FATAL: G1a is bit-identical to G0 -- the ranking term did not reach the weights")
    sys.exit(1)
print("  OK: the ranking term changed the weights")
PY

qlog "==================== P2b DONE failures=$QFAIL ===================="
