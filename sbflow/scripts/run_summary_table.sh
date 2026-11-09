#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
echo "=== generated table (preview) ==="
python3 -u -m tools.v6_24h.make_summary_table
