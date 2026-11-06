#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE/runs/12h/anchors || exit 1
ls -la base_ep10_750_512 | awk '{print $5, $9}'
echo ====
ls -la base_ep10_600 | awk '{print $5, $9}'
