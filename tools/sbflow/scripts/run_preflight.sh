#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
python3 -u -m sbflow.preflight \
  --cache-a runs/blog_sbflow/cache/Impressionism.pt \
  --cache-b runs/blog_sbflow/cache/Ukiyo_e.pt \
  --repo-root . --n 3
