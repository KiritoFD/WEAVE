#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 -u tools/pseudo_gt/build_assets.py --stage retrieval
echo EXIT=$?
