#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/blog_sbflow/v11/logs
python3 -u -m sbflow.p2_edm_control --threads 4 2>&1 | tee runs/blog_sbflow/v11/logs/p2.log
