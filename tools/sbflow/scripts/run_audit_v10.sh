#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
python3 -u -m sbflow.audit_v10 --ckpt runs/blog_sbflow/v10/exp_04L/model_step6000.pt \
  --steps 400 --batch 1024
