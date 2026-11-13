#!/bin/bash
# 12GB smoke gate for StyleGallery: 12 optimize steps, 1 iteration.
# NOT a quality result — a VRAM/functionality check only.
# Usage: bash run_stylegallery_smoke_12g.sh <content.png> <style.png> [out_dir]
set -euo pipefail

CONTENT="${1:?content image required}"
STYLE="${2:?style image required}"
OUT="${3:-outputs/sg12_smoke}"
REPO="$(cd "$(dirname "$0")/../../.." 2>/dev/null && pwd)/StyleGallery"
REPO="${STYLEGALLERY_REPO:-$HOME/StyleGallery}"
ENV_DIR="${SG12_ENV:-$HOME/weave_baselines/sg12env}"

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
cd "$REPO"
mkdir -p "$OUT"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
python demo_accelerate.py \
  --mode lcm \
  --model_name pretrained_models/runwayml_stable-diffusion-v1-5 \
  --lora_path pretrained_models/lcm-lora-sd1.5/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors \
  --content_image "$CONTENT" \
  --style_images "$STYLE" \
  --output_folder "$OUT" \
  --noise_steps 15 --num_optimize_steps 12 --iters 1 \
  --lr 0.25 --c_ratio 0.33 --seed 42 \
  ${STYLEGALLERY_DINO_CPU:+${STYLEGALLERY_DINO_CPU:+}}

echo "Smoke output: $OUT/result_lcm.png"
ls -la "$OUT"/*.png 2>/dev/null || true
