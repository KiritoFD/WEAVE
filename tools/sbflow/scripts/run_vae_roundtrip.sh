#!/usr/bin/env bash
# Run the VAE round-trip control. This needs the GPU, so it is only launched when the
# 24h queue is not holding it.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then
  echo "GPU busy (pid $BUSY); refusing to run the control concurrently."
  exit 2
fi
python3 -u -m tools.v6_24h.vae_roundtrip_control
