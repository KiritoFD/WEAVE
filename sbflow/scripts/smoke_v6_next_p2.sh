#!/usr/bin/env bash
# P2 smoke: 4 steps of each arm to verify --init-from, the frozen critic, and the step-1 gradient
# ratio print before committing 36 minutes of GPU time.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi

INIT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
CRITIC=runs/v6_next/probes/r1_r1.pt
COMMON="--mode direct --base-dim 16 --steps 4 --microbatch 80 --accum 1 --seed 42 \
--canon-mode c1_gentle --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 \
--ref-top-k 1024 --init-from $INIT"

FAIL=0
for spec in "G0_smoke:" "G1a_smoke:--rank-critic $CRITIC --rank-lambda 0.1 --rank-tau 0.5" \
            "G1b_smoke:--rank-critic $CRITIC --rank-lambda 0.3 --rank-tau 0.5"; do
  name=${spec%%:*}; extra=${spec#*:}
  echo "########## $name ##########"
  if ! python3 -u tools/v6/train_eval_v6.py --name "$name" $COMMON $extra 2>&1 | tail -n 8; then
    echo "  >>> FAILED $name"; FAIL=$((FAIL+1))
  fi
done
echo
if [ "$FAIL" -eq 0 ]; then echo "P2 SMOKE PASSED"; else echo "P2 SMOKE FAILED ($FAIL)"; fi
exit "$FAIL"
