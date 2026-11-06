#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "--- train proc ---"; pgrep -af 'run.py' || echo none
echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- train.log tail ---"; tail -n 30 runs/v4/S1_A_P5/train.log 2>/dev/null | grep -v 'it/s]' | cut -c1-260
echo "--- res.log peak ---"; awk -F'vram_mib=' '{split($2,a," "); if(a[1]+0>vm)vm=a[1]+0} END{print "peak_vram="vm" mib"}' runs/v4/S1_A_P5/res.log 2>/dev/null
echo "--- ckpts ---"; ls -1 runs/v4/S1_A_P5 2>/dev/null | tr '\n' ' '
echo; echo "--- progress ---"; tail -n 3 runs/gt/PROGRESS.md | cut -c1-200
