#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
mkdir -p deliver
python3 -m tools.v6_next.delivery_table > deliver/V6_RESULTS_TABLE.md 2>&1
echo "wrote deliver/V6_RESULTS_TABLE.md ($(wc -l < deliver/V6_RESULTS_TABLE.md) lines)"
head -n 24 deliver/V6_RESULTS_TABLE.md
