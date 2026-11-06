#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE/runs/12h/anchors || exit 1
ls -la | awk '{print $5, $9}'
