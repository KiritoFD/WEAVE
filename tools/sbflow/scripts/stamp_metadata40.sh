#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
SHA=${1:?sha required}
python3 -u sbflow/stamp_metadata40.py --sha "$SHA"
