#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
cat /mnt/i/Latent_Style/pseudo_gt_24h/eval_render/S2_E7_final/v4_eval750.json
echo "== S2 gate =="
grep -a "Internal dynamics" runs/v4/S2_E7/train.log
echo "== S2 train res =="
grep -a -E "Reached stop|Internal early stop summary" runs/v4/S2_E7/train.log
echo "== S2 render =="
grep -a "done" runs/v4/S2_E7/render.log | tail -n 2
