#!/usr/bin/env bash
# V4 全量 750 all-pairs 评测（protocol paper / 512 / CLIP-B32 / DINOv2-s / b8 / TRT fp16）。
# Usage: v4_full_eval.sh <name> <ckpt> <out_root> <steps> <strength> <cfg> [seed]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export HF_HOME="${WEAVE_EVAL_CACHE_ROOT:-$ROOT/../eval_cache}/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
name="$1"; ckpt="$2"; root="$3"; steps="$4"; strength="$5"; cfg="$6"; seed="${7:-0}"
out="$root/$name"; mkdir -p "$out"
echo "ckpt=$ckpt steps=$steps strength=$strength cfg=$cfg seed=$seed batch=8 pairs=750" > "$out/v4_run_meta.txt"
V4_NUM_STEPS="$steps" V4_EDIT_STRENGTH="$strength" V4_CFG="$cfg" V4_SEED="$seed" \
python3 tools/12h/eval_fast_trt_fp16.py --protocol paper --checkpoint "$ckpt" --output "$out" \
  --device cuda --batch 8 --metric-batch 32 --decode-batch 8 --image-size 512 --workers 4 \
  --no-save-images --gen-dtype fp32 --vae-dtype fp32 --sync-timing --decoder trt \
  --num-steps "$steps" --seed "$seed" --backbone "v4_$name" > "$out/eval.log" 2>&1
echo "EXIT=$?" >> "$out/eval.log"
