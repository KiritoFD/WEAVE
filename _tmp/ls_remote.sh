#!/usr/bin/env bash
cd /mnt/i/Latent_Style || exit 1
for d in WEAVE/runs/p1 WEAVE/runs/p2 WEAVE/runs/p3 WEAVE/runs/v4 WEAVE/runs/pgt pseudo_gt_24h; do
  echo "==== $d"
  ls -la "$d" 2>/dev/null | awk '{print $5, $9}'
done
echo "==== pseudo_gt_24h subdirs"
ls -la pseudo_gt_24h 2>/dev/null | awk '{print $5, $9}'
