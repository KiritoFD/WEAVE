#!/usr/bin/env bash
# Generate the visual grid for human blind rating. Needs the GPU.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then
  echo "GPU busy (pid $BUSY); refusing to run."
  exit 2
fi
CKPT=${1:-runs/v6/N7_dim24/N7_dim24.pt}
python3 -u -m tools.v6_24h.make_grid --ckpt "$CKPT" --n-sources "${2:-4}"
