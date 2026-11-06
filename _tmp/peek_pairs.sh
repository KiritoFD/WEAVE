#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
date -u '+%F %T UTC'
echo "--- covfix tmux ---"
tmux ls 2>&1 | head -5
echo "--- covfix log ---"
tail -n 8 runs/p2/stage_covfix.log 2>/dev/null
echo "--- pairs tree ---"
ls -la runs/pgt/pairs 2>/dev/null | head -20
for d in runs/pgt/pairs/*/; do echo "$d :: $(ls "$d" | head -6 | tr '\n' ' ')"; done 2>/dev/null
echo "--- manifests ---"
ls -la runs/pgt/pairs/*/*.json 2>/dev/null | head -12
echo "--- gpu ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
free -g | head -2
