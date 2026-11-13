#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE
export HF_HUB_OFFLINE=1
python3 -u -m tools.v6_next.w_paired_compare > runs/v6_w/w_paired_report.txt 2>&1
python3 -u -m tools.v6_next.w_select_best
