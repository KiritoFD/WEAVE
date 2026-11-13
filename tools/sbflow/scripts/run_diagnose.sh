#!/usr/bin/env bash
# Run selected V9 diagnostic experiments on one checkpoint.
# Args: TAG  CKPT  EXPERIMENTS...
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
TAG=${1:?tag}; CKPT=${2:?ckpt}; shift 2
python3 -u -m sbflow.diagnose "$@" --ckpt "$CKPT" --tag "$TAG" --repo-root . --weights raw
