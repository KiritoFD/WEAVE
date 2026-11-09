#!/usr/bin/env bash
# P0: real-batch validation of the 24h code changes.
#  1) A0 equivalence: the refactored decoupled loss must reproduce A0's step-1 loss
#     exactly (1.3165 from the historical A0_ref run).
#  2) every new flag must run 2 steps without crashing and must print its diagnostics.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/v6_24h/_p0
mkdir -p "$OUT/logs"

BASE="--mode direct --canon-mode c1_gentle --base-dim 48 --steps 2 --microbatch 80 \
--accum 3 --lr 3e-4 --coherence-beta 0.0 --w-rec 1.0 --w-style-dist 0.25 --w-cross 0.35 \
--w-canon-inv 1.5 --seed 42"

run() {
  local NAME=$1; shift
  echo "########## $NAME ##########"
  python3 -u tools/v6/train_eval_v6.py --name "$NAME" $BASE "$@" \
    > "$OUT/logs/${NAME}.log" 2>&1
  local rc=$?
  grep -aE "params=|B1 branch|step1 diag|reference pool|step 1/2|Error|Traceback" \
    "$OUT/logs/${NAME}.log" | head -8
  if [ $rc -ne 0 ]; then echo "  RC=$rc  <-- FAILED"; tail -6 "$OUT/logs/${NAME}.log"; fi
  return 0
}

run P0_a0_equiv
run P0_no_style      --no-style
run P0_no_canon      --canon-mode none
run P0_dim32         --base-dim 32
run P0_inv_off       --lambda-cross 0.35 --lambda-canon 0.0
run P0_no_b1         --no-b1
run P0_same_off      --no-same-style
run P0_core_off      --core-sampler-off
run P0_ref_k256      --ref-top-k 256
run P0_no_inject_64  --no-inject-64
run P0_no_inject_32  --no-inject-32
run P0_no_inject_16  --no-inject-16

echo "########## P0 REAL-BATCH CHECKS DONE ##########"
