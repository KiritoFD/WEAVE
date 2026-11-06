#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "TIME $(date '+%F %T')"
tail -n 3 runs/v4/build_reverse_e3.log 2>/dev/null | cut -c1-200
ls -1 ../pseudo_gt_24h/pairs/reverse_e3 2>/dev/null | wc -l
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
free -m | awk '/Mem:/{print "RAM",$3"/"$2}'
tail -n 2 runs/gt/PROGRESS.md
