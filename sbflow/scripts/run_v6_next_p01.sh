#!/usr/bin/env bash
# P0.1: complete L11's independent evaluation on Heldout400 and Same150 so the 218K baseline is
# not chosen on Val200 alone.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi

CKPT=runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
for tier in heldout400 same150; do
  echo "########## L11 on $tier ##########"
  python3 -u -m tools.v6_24h.eval_tiers --ckpt "$CKPT" \
    --tasks "runs/v6_24h/${tier}_tasks.json" --out "runs/v6_24h/eval_${tier}_L11"
done
