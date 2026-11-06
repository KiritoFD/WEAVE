#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 -u _tmp/inspect_ret.py
echo EXIT=$?
