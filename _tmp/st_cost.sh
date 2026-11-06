#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "=== pairs_base.log tail"
tail -n 6 runs/p2/pairs_base.log 2>/dev/null
echo "=== pairs_positive.log tail"
tail -n 6 runs/p2/pairs_positive.log 2>/dev/null
echo "=== render logs"
for f in runs/p2/render_A_D_r0.log runs/pgt/S1_positive_b256/render.log runs/pgt/S2_reverse_b256/render.log; do echo "-- $f"; tail -n 5 "$f" 2>/dev/null; done
echo "=== metrics logs"
tail -n 4 runs/p2/metrics_S0_base_b256.log 2>/dev/null
echo "=== render output dirs"
ls -1 runs/p2/canonical 2>/dev/null | head -20
echo "=== 600/150 sets"
ls -1 /mnt/i/Latent_Style/pseudo_gt_24h 2>/dev/null
