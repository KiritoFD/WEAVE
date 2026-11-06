#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE/runs/v4 || exit 1
ls -la *.log *.txt 2>/dev/null | awk '{print $5, $9}'
