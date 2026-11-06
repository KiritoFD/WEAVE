#!/bin/bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "== all python =="
ps aux | grep -E "python3" | grep -v grep | head -6
echo "== prof log size =="
ls -l /mnt/i/Latent_Style/WEAVE/_tmp/prof_render.log
