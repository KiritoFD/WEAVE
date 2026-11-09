#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
echo "=== smoke16 ==="
python3 -m tools.v6_24h.smoke16
echo "=== queue list ==="
python3 -m tools.v6_24h.queue --list | tail -5
