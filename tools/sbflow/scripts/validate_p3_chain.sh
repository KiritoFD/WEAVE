#!/usr/bin/env bash
# Validate the full P3 chain on the GPU box before spending 4000 updates:
# tiny pivot training -> unconditional sampling -> scoring.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TMP=runs/blog_sbflow/v11/_p3_smoke
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"
rm -rf "$TMP"; mkdir -p "$TMP"

echo "########## 1/3 tiny pivot training ##########"
python3 -u -m sbflow.train5 --config configs/v11/p3.json --caches $C \
  --steps 4 --batch 64 --width 32 --depth 1 --heads 2 --warmup 2 \
  --log-interval 1 --save-interval 4 --save-equiv "" --full-equiv 0 \
  --output "$TMP/train" 2>&1 | tail -5

echo "########## 2/3 unconditional sampling ##########"
python3 -u -m sbflow.gen_uncond --ckpt "$TMP/train/model_step4.pt" \
  --out "$TMP/gen" --per-style 2 --steps 2 --batch 2 2>&1 | tail -6

echo "########## 3/3 scoring ##########"
python3 -u -m sbflow.score_uncond --gen "$TMP/gen" --out "$TMP/score" --threads 4 \
  2>&1 | tail -12

echo "########## P3 CHAIN VALIDATION DONE ##########"
