#!/usr/bin/env bash
# Verify M0 imports and CLI without touching the GPU-heavy path.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
python3 -m tools.v6_24h.m0_alpha --help
