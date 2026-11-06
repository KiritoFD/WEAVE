#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE/runs/p2 || exit 1
ls -la | awk '{print $5, $9}'
