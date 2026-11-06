#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE/runs/v4 || exit 1
for d in prior prior_run1_earlystop_cfg diag p0 arms style_breakthrough smoke; do echo "== $d"; ls -la "$d" 2>/dev/null | awk '{print $5, $9}'; done
