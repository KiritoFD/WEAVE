#!/usr/bin/env bash
ls -la /mnt/i/Latent_Style/WEAVE/runs/trt/engines | awk '{print $5, $9}'
echo ====
ls -la /mnt/i/Latent_Style/WEAVE/runs/trt | awk '{print $5, $9}'
