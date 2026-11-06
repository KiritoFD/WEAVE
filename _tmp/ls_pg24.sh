#!/usr/bin/env bash
cd /mnt/i/Latent_Style/pseudo_gt_24h || exit 1
ls -la | awk '{print $5, $9}'
echo "==== eval_render"
ls -la eval_render 2>/dev/null | awk '{print $5, $9}'
echo "==== bench_render"
ls -la bench_render 2>/dev/null | awk '{print $5, $9}'
