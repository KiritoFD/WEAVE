#!/usr/bin/env bash
set -uo pipefail
pkill -f build_baseline_assets.py || true
sleep 1
echo "--- remaining ---"
ps -eo pid,etime,args | grep build_baseline | grep -v grep || echo NONE
echo "--- stage dir ---"
du -sh /mnt/i/Latent_Style/WEAVE/_baseline_stage 2>/dev/null || echo "no stage dir"
