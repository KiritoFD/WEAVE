#!/usr/bin/env bash
# P1 R1: train the latent style discriminator and report the classification gate.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
if [ -n "$BUSY" ]; then echo "GPU busy (pid $BUSY)"; exit 2; fi
TAG=${1:-r1}
STEPS=${2:-3000}
shift 2 2>/dev/null || shift $#
python3 -u -m tools.v6_next.probe_style --tag "$TAG" --steps "$STEPS" "$@"
