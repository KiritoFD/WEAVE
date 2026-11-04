#!/usr/bin/env bash
# Same as eval_path_trt.sh plus an optional config-override (attribution runs).
# Usage: eval_path_attr.sh <tag> <ckpt> <out_dir> <batch> <override.json>
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export HF_HOME="${WEAVE_EVAL_CACHE_ROOT:-$ROOT/../eval_cache}/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tag="$1"; ckpt="$2"; out="$3"; batch="$4"; override="$5"
mkdir -p "$out"
python3 tools/12h/eval_fast_trt_fp16.py --protocol paper --checkpoint "$ckpt" --output "$out" \
  --device cuda --batch "$batch" --metric-batch 32 --decode-batch "$batch" \
  --workers 4 --no-save-images --gen-dtype fp32 --vae-dtype fp32 --sync-timing \
  --backbone "$tag" --decoder trt --config-override "$override" \
  > "$out/eval.log" 2>&1
echo "EXIT=$?" >> "$out/eval.log"
date >> "$out/eval.log"

