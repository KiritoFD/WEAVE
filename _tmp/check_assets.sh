#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "=== p0 refs ==="
ls -la runs/p0/reference_selection.json 2>/dev/null
echo "=== p2_render A ==="
ls /mnt/i/Latent_Style/eval_assets/p2_render/A/images 2>/dev/null | wc -l
ls -la /mnt/i/Latent_Style/eval_assets/p2_render/A/metrics.csv 2>/dev/null
echo "=== p2_render cells ==="
ls /mnt/i/Latent_Style/eval_assets/p2_render 2>/dev/null
echo "=== old ckpt ==="
ls -la exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt 2>/dev/null
echo "=== runs ==="
ls runs 2>/dev/null
echo "=== free disk ==="
df -h /mnt/i | tail -1
