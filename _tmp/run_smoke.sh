#!/usr/bin/env bash
# One-off wiring smoke run (5 steps, batch 32, early stop/probe off).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u run.py --config configs/pgt/S0_smoke.json > runs/pgt/_smoke.log 2>&1
rc=$?
echo "--- smoke log tail ---"
tail -n 45 runs/pgt/_smoke.log
echo "SMOKE rc=$rc"
exit $rc
