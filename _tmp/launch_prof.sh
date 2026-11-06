#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
nohup bash _tmp/prof_render.sh > _tmp/prof_render.log 2>&1 &
echo "pid=$!"
sleep 3
