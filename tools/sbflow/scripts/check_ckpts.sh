#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== checkpoints referenced by the post-queue P5 step ==="
for p in runs/v6/L7_highC_dim32/L7_highC_dim32.pt \
         runs/v6/L4_ref_full/L4_ref_full.pt \
         runs/v6/L2_highS/L2_highS.pt \
         runs/v6/N3_dim32/N3_dim32.pt; do
  if [ -f "$p" ]; then
    echo "  OK   $p  ($(du -h "$p" | cut -f1))"
  else
    echo "  MISS $p"
  fi
done
