#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "TIME $(date '+%F %T')"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
free -m | awk '/Mem:/{print "RAM used/total",$3"/"$2" MiB"}'
tail -n 8 runs/gt/PROGRESS.md
echo "--- build_e3 log ---"; tail -n 6 runs/v4/build_reverse_e3.log 2>/dev/null | cut -c1-220
echo "--- e3 cache ---"; ls -la ../pseudo_gt_24h/pairs/reverse_e3 2>/dev/null | tail -4
