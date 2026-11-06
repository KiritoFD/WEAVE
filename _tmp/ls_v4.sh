#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE/runs/v4 || exit 1
for d in S1_A_P5 S2_E7 arms diag p0 prior; do
  echo "==== $d"
  ls -la "$d" 2>/dev/null | awk '{print $5, $9}'
done
