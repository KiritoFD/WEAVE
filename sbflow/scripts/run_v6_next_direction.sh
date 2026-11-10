#!/usr/bin/env bash
# P0.3 geometric direction diagnosis on the three baselines.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi

python3 -u -m tools.v6_next.eval_direction \
  --ckpt runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt --tag L11 "$@"
python3 -u -m tools.v6_next.eval_direction \
  --ckpt runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt --tag L8 "$@"
