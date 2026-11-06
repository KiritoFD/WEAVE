#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
for d in runs/p1 runs/p2 runs/p3 runs/p4 runs/p5 runs/12h runs/audit runs/infra; do
  echo "== $d =="
  ls "$d" 2>/dev/null | head -40
done
echo "== docs =="
ls docs | head -20
echo "== pseudo_gt_24h =="
ls /mnt/i/Latent_Style/pseudo_gt_24h | head -30
