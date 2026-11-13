#!/bin/bash
# StyleGallery 12GB setup: env + weights + patch. Run on the RTX 3060 WSL host.
# Usage: bash setup_stylegallery_12g.sh [StyleGallery_clone_dir] [work_root]
#   default clone dir: ~/StyleGallery ; weights land in <clone>/pretrained_models/
set -euo pipefail

SG_REPO="${1:-$HOME/StyleGallery}"
WORK="${2:-$HOME/weave_baselines}"
ENV_DIR="$WORK/sg12env"
mkdir -p "$WORK"

# ---------------------------------------------------------------- env
if [ ! -d "$ENV_DIR" ]; then
  python3 -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -V
pip install -q --upgrade pip
# Torch 2.6 cu126 to match the author's environment; diffusers pinned per doc.
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q "diffusers==0.36.0" "transformers>=4.46,<5" accelerate safetensors \
  Pillow numpy scipy scikit-learn opencv-python-headless tqdm pyyaml \
  "huggingface_hub>=0.34" sentencepiece protobuf
pip freeze > "$WORK/sg12_requirements.lock"

# ---------------------------------------------------------------- repo
if [ ! -d "$SG_REPO/.git" ]; then
  git clone https://github.com/iiiiiiiword/StyleGallery "$SG_REPO"
fi
cd "$SG_REPO"
git log -1 --format='%H %ci' | tee "$WORK/stylegallery_commit.txt"
python "$WORK/patch_stylegallery_12g.py" --repo "$SG_REPO" 2>/dev/null \
  || python "$(dirname "$0")/patch_stylegallery_12g.py" --repo "$SG_REPO"

# ---------------------------------------------------------------- weights
# HF_HOME kept inside the repo so paths stay relative and self-contained.
export HF_HOME="$SG_REPO/.hf"
mkdir -p pretrained_models

# SD1.5 diffusers layout (demo expects .../runwayml_stable-diffusion-v1-5)
if [ ! -f pretrained_models/runwayml_stable-diffusion-v1-5/model_index.json ]; then
  hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir pretrained_models/runwayml_stable-diffusion-v1-5
fi
# DINOv2-B for semantic matching
if [ ! -f pretrained_models/facebook/dinov2-base/model.safetensors ] \
   && [ ! -f pretrained_models/facebook/dinov2-base/pytorch_model.bin ]; then
  hf download facebook/dinov2-base \
    --local-dir pretrained_models/facebook/dinov2-base
fi
# LCM LoRA (primary fast path)
if ! ls pretrained_models/lcm-lora-sd1.5/*.safetensors >/dev/null 2>&1; then
  hf download latent-consistency/lcm-lora-sdv1-5 \
    --local-dir pretrained_models/lcm-lora-sd1.5
fi
# Optional Hyper-SD (only if explicitly wanted; ~700MB)
if [ "${WITH_HYPER_SD:-0}" = "1" ]; then
  hf download ByteDance/Hyper-SD Hyper-SD15-4steps-lora.safetensors \
    --local-dir pretrained_models/Hyper-SD
fi

echo "SETUP DONE. Next: 12GB smoke gate (see WEAVE_ICME_12GB_BASELINE_REPRO.md section 1.5)"
