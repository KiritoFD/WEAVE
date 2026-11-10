#!/usr/bin/env bash
# P1 smoke: 4 steps of B1 to verify the teacher, the L_distill magnitude, and that the distillation
# gradient reaches the weights (the fp16-underflow class of bug must not recur; we run bf16).
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi

Q2A=runs/v6/Q2a_embed_style/Q2a_embed_style.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
C="--mode direct --base-dim 16 --steps 4 --microbatch 40 --accum 1 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --amp-dtype bf16 --init-from $Q2A"

echo "########## smoke control (no distill) ##########"
python3 -u tools/v6/train_eval_v6.py --name S_distill_ctrl $C 2>&1 | grep -aE "Eval200|Error" | head -2

echo "########## smoke distill lam=1.0 ##########"
python3 -u tools/v6/train_eval_v6.py --name S_distill_on $C \
  --distill-teacher "$Q2A" --distill-critic "$CRITIC" --distill-steps 1 \
  --distill-rms 0.25 --distill-lambda 1.0 2>&1 \
  | grep -aE "DISTILL|distill teacher|Eval200|Error" | head -5

python3 - <<'PY'
import torch
from pathlib import Path
def m(n):
    return torch.load(f"runs/v6/{n}/{n}.pt", map_location="cpu", weights_only=False)["model"]
a, b = m("S_distill_ctrl"), m("S_distill_on")
mx = max(float((a[k].float()-b[k].float()).abs().max()) for k in a)
nd = sum(1 for k in a if float((a[k].float()-b[k].float()).abs().max()) > 0)
print(f"\n  distill vs control: max|dW|={mx:.3e} {nd}/{len(a)} -> "
      + ("FATAL: distillation did nothing" if mx == 0 else "distillation reached the weights"))
PY
