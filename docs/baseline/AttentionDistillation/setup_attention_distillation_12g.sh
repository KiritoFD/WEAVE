#!/bin/bash
# Attention Distillation 12GB setup: env + SD1.5 weights + repo clone.
# Usage: bash setup_attention_distillation_12g.sh <repo_dir> <env_dir> [--sd15 <existing_sd15_dir>]
set -euo pipefail

REPO_DIR="${1:?repo dir required}"
ENV_DIR="${2:?env dir required}"
SD15_LINK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sd15) SD15_LINK="$2"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$ENV_DIR"

if [ ! -d "$ENV_DIR" ]; then
  python3 -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q "diffusers==0.36.0" "transformers>=4.46,<5" accelerate safetensors \
  Pillow numpy scipy opencv-python-headless tqdm pyyaml "huggingface_hub>=0.34" \
  sentencepiece protobuf gradio  # gradio only if webui path is used
pip freeze > "$ENV_DIR/requirements.lock"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/xugao97/AttentionDistillation "$REPO_DIR"
fi
cd "$REPO_DIR"
git log -1 --format='%H %ci' | tee "$ENV_DIR/ad_commit.txt"

# SD1.5 weights: reuse an existing diffusers-format dir when provided
if [ -n "$SD15_LINK" ]; then
  mkdir -p pretrained_models
  if [ ! -e pretrained_models/stable-diffusion-v1-5 ]; then
    ln -s "$SD15_LINK" pretrained_models/stable-diffusion-v1-5
  fi
fi
if [ ! -f pretrained_models/stable-diffusion-v1-5/model_index.json ]; then
  export HF_HOME="$REPO_DIR/.hf"
  hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir pretrained_models/stable-diffusion-v1-5
fi
echo "AD SETUP DONE"
