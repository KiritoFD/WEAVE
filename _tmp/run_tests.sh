#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 -m pytest tests/test_pseudo_gt_patterns.py -q 2>&1 | tail -6
