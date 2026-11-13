#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
echo "=== S6 gate on CPU ==="
python3 -m tools.v6_24h.test_s6 --device cpu --batch 2
