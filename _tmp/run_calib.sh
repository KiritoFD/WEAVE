#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 -u _tmp/calib_match.py
echo EXIT=$?
