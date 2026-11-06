#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
nohup python3 tools/pseudo_gt/build_assets.py --stage dino > runs/p2/pgt_assets_dino.log 2>&1 &
echo STARTED
