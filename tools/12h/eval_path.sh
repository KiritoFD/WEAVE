#!/usr/bin/env bash
# Unified eval runner for the 12h analysis round.
# Usage: eval_path.sh <tag> <ckpt> <out_dir> <batch> <decoder:torch|trt>
# Protocol is frozen to "paper" (750 all-pairs, 256px, 8 steps, AdaIN 2.0).
# Only the generation/decode batch and the VAE decoder backend vary.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export HF_HOME="${WEAVE_EVAL_CACHE_ROOT:-$ROOT/../eval_cache}/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tag="$1"; ckpt="$2"; out="$3"; batch="$4"; decoder="$5"
mkdir -p "$out"
python3 tools/eval_fast.py --protocol paper --checkpoint "$ckpt" --output "$out" \
  --device cuda --batch "$batch" --metric-batch 32 --decode-batch "$batch" \
  --workers 4 --no-save-images --gen-dtype fp32 --vae-dtype fp32 --sync-timing \
  --backbone "$tag" --decoder "$decoder" \
  > "$out/eval.log" 2>&1
echo "EXIT=$?" >> "$out/eval.log"
date >> "$out/eval.log"

