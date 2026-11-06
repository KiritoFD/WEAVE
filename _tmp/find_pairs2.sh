#!/usr/bin/env bash
cd /mnt/i/Latent_Style/pseudo_gt_24h || exit 1
find . -maxdepth 3 \( -iname "*manifest*" -o -iname "*pairs*" -o -iname "*cache*" \) | head -40
echo "==== top"
ls -la | awk '{print $5, $9}' | head -40
