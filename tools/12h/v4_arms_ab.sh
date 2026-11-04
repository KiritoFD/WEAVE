#!/usr/bin/env bash
# A/B 两臂：batch 256 / accum 1 / lr 2.8e-4 / 6000 step（各 1.536M 样本，等预算）。
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
mkdir -p runs/v4/arms
exec > runs/v4/arms/arms_AB.log 2>&1
for arm in A B; do
  python3 -m v4.train --config "v4/arms/$arm.json" --run-name "$arm" \
    > "runs/v4/arms/${arm}_train.log" 2>&1
  echo "$arm EXIT=$? $(date +%H:%M:%S)"
done
echo ARMS_AB_DONE
