#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== S2 epochs =="
grep -a "Internal dynamics" runs/v4/S2_E7/train.log | tail -n 4
echo "== S2 stop =="
grep -a -E "Internal early stop summary|Reached stop_after|Saved milestone" runs/v4/S2_E7/train.log | tail -n 3
echo "== gpu =="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "== progress =="
tail -n 2 runs/gt/PROGRESS.md
