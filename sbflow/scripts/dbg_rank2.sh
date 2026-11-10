#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
python3 -u tools/v6_next/diag_rank_grad2.py 2>&1 | grep -avE "^Loading weights|it/s\]" | tail -n 30
