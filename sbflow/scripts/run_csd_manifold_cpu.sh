#!/usr/bin/env bash
# CPU-only part: real-source population statistics. Safe to run while the queue owns the GPU.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 -u -m tools.v6_24h.csd_manifold_check
