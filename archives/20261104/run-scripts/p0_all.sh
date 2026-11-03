#!/usr/bin/env bash
# P0 diagnostics, strictly sequential on the single GPU.
set -u
R=/mnt/i/Latent_Style/runs/p0_diag
mkdir -p "$R"
bash /mnt/i/Latent_Style/runs/p0_adain.sh
bash /mnt/i/Latent_Style/runs/p0_capacity.sh
bash /mnt/i/Latent_Style/runs/p0_curve.sh
echo P0_ALL_DONE | tee -a "$R/all.log"
