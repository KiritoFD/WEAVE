#!/usr/bin/env bash
# Q3: independent evaluation of the Val200-selected candidates.
#
# Selection is on Val200 ONLY, as the plan requires. Heldout400 is used here for confirmation and
# this run is its recorded use. Two candidates: Q2a (embedding-space supervision, the Val200 winner)
# and G1b (logit-margin ranking only, the previous best) so the mechanism difference is confirmed on
# source-disjoint data rather than assumed from Val200.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_24h/queue_q3.log
source sbflow/scripts/q_lib.sh

wait_for_gpu 240 || exit 3

confirm () {
  local name=$1
  local ck="runs/v6/${name}/${name}.pt"
  [ -s "$ck" ] || { qlog "Q3: $name checkpoint missing"; return 1; }
  for tier in heldout400 same150; do
    run_step "Q3_${name}_${tier}" "runs/v6_next/eval_${tier}_${name}/eval.json" \
      python3 -u -m tools.v6_24h.eval_tiers --ckpt "$ck" \
        --tasks "runs/v6_24h/${tier}_tasks.json" \
        --out "runs/v6_next/eval_${tier}_${name}"
  done
}

# Val200 winner, then the mechanism-only baseline
confirm Q2a_embed_style
confirm P2_G1b_bf16

qlog "==================== Q3 DONE failures=$QFAIL ===================="
