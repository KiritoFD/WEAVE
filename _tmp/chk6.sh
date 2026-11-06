#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== engines =="
ls runs/trt/engines/
echo "== S2 =="
grep -a "Internal dynamics" runs/v4/S2_E7/train.log | tail -n 2
grep -a -E "Internal early stop summary|Reached stop_after|Saved milestone" runs/v4/S2_E7/train.log | tail -n 2
echo "== progress =="
tail -n 2 runs/gt/PROGRESS.md
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
