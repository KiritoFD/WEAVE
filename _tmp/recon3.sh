#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "=== S2 RUN DIR ==="; ls -la runs/v4/S2_E7
echo "=== S2 LOG TAIL ==="; tail -60 runs/v4/S2_E7/*.log 2>/dev/null
echo "=== S2 EVAL JSON ==="; cat ../pseudo_gt_24h/eval_render/S2_E7_final/v4_eval750.json
echo "=== CONFIG S2 ==="; cat configs/pgt/S2_reverse.json
echo "=== CONFIG S1 ==="; cat configs/pgt/S1_positive.json
echo "=== TRAIN CACHE ROOTS ==="; ls -d ../pseudo_gt_24h/* 2>/dev/null
echo "=== CACHE SUB ==="; find ../pseudo_gt_24h -maxdepth 3 -type d -name 'cache*' 2>/dev/null | head -20
echo "=== S1 RUN DIR ==="; ls -la runs/v4/S1_A_P5 | head -20
