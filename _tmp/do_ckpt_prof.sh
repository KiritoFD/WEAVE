#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 _tmp/ckpt_same.py
nohup bash _tmp/prof_render.sh > _tmp/prof_render.log 2>&1 &
echo "prof_pid=$!"
