#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== log =="
cat _tmp/prof_render.log
echo "== proc =="
ps aux | grep -E "prof_render" | grep -v grep | head -2
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
