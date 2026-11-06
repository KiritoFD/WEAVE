#!/bin/bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
nohup python3 tools/pseudo_gt/build_assets.py --stage latents > runs/p2/pgt_assets_latents.log 2>&1 &
echo "started $!"
