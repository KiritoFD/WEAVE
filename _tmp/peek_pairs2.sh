#!/usr/bin/env bash
A=/mnt/i/Latent_Style/pseudo_gt_24h
echo "--- pairs dirs ---"
ls -la "$A/pairs" 2>/dev/null
echo "--- each dir ---"
for d in "$A"/pairs/*/; do
  n=$(ls "$d" 2>/dev/null | wc -l)
  echo "$(basename "$d") files=$n :: $(ls "$d" 2>/dev/null | head -5 | tr '\n' ' ')"
done
echo "--- manifests ---"
ls -la "$A"/pairs/*/*.json 2>/dev/null | head -20
echo "--- covfix ---"
tail -n 6 /mnt/i/Latent_Style/WEAVE/runs/p2/stage_covfix.log 2>/dev/null
tail -n 3 /mnt/i/Latent_Style/WEAVE/runs/gt/PROGRESS.md
