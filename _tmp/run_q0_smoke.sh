#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
python3 -u tools/pseudo_gt/q0_quality.py --limit 2 --out /tmp/pgt_q0_smoke.json
echo EXIT=$?
