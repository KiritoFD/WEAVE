#!/usr/bin/env bash
# Benchmark the SD-VAE five-style trainer at batch 1024 for several patch sizes.
# 64x64 latent -> patch 2 = 1024 tokens, patch 4 = 256, patch 8 = 64.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"

for P in 8 4 2; do
  echo "=== patch=$P tokens=$(( (64 / P) * (64 / P) )) ==="
  rm -rf runs/blog_sbflow/v10/_bench_p$P
  python3 -u -m sbflow.train5 --experiment-id "bench_p$P" --caches $C \
    --output "runs/blog_sbflow/v10/_bench_p$P" \
    --steps 12 --batch 1024 --patch $P --width 256 --depth 6 --heads 4 \
    --warmup 2 --log-interval 4 --save-interval 100 2>&1 | \
    grep -E "params=|step=12/12|CUDA out of memory" || true
done
echo "=== BENCH DONE ==="
