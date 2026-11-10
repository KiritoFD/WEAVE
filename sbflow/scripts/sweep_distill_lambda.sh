#!/usr/bin/env bash
# Can the student be FORCED to fit the teacher? Sweep the distillation weight.
#
# The 800-step run at lambda=1.0 left the student moving only ~1% of the teacher's refine
# displacement (rel_move 0.012 vs the control's 0.026), i.e. it did not fit. Two explanations remain:
# the loss weight was too small relative to the V6 objective, or the operator cannot represent the
# edit. A weight sweep at short length separates them: if a larger lambda makes rel_move climb toward
# the teacher, the weight was the problem.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi

Q2A=runs/v6/Q2a_embed_style/Q2a_embed_style.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
C="--mode direct --base-dim 16 --steps 200 --microbatch 80 --accum 3 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --amp-dtype bf16 --init-from $Q2A"

for LAM in 1 5 20; do
  echo "########## lambda=$LAM ##########"
  python3 -u tools/v6/train_eval_v6.py --name "F_lam${LAM}" $C \
    --distill-teacher "$Q2A" --distill-critic "$CRITIC" --distill-steps 1 \
    --distill-rms 0.25 --distill-lambda "$LAM" 2>&1 \
    | grep -aE "DISTILL|Eval200|Error" | head -3
done

echo "########## fit error for each lambda ##########"
python3 -u -m tools.v6_next.fit_error --per-style 100 \
  --students "F_lam1,F_lam5,F_lam20" 2>&1 | tail -n 12
