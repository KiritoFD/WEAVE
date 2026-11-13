#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 -u -m sbflow.scripts.profile_ot_sampler
