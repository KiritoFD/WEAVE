#!/usr/bin/env bash
# 独立参考池（author-held-out validation，非 data/test）风格评估。
# 40 对（每风格 8 张内容图）保留 PNG，再用 v4_refpool_eval.py 对新参考池打分。
# Usage: v4_refpool_gen.sh <ckpt> <root_out> [steps=20]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export HF_HOME="${WEAVE_EVAL_CACHE_ROOT:-$ROOT/../eval_cache}/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ckpt="$1"; root="$2"; steps="${3:-20}"
mkdir -p "$root"
exec >> "$root/refpool.log" 2>&1

evalrun() {
  name="$1"; out="$root/$name"; mkdir -p "$out"
  python3 tools/12h/eval_fast_trt_fp16.py --protocol paper --checkpoint "$ckpt" --output "$out" \
    --device cuda --batch 8 --metric-batch 32 --decode-batch 8 --image-size 512 --workers 4 \
    --gen-dtype fp32 --vae-dtype fp32 --sync-timing --decoder trt \
    --max-src-samples 8 --num-steps "$steps" --seed 0 --backbone "v4_$name" \
    > "$out/eval.log" 2>&1
  echo "EXIT=$?" >> "$out/eval.log"
  echo "$name gen done $(date +%H:%M:%S)"
  python3 tools/12h/v4_refpool_eval.py "$out" --tag "$name" > "$out/refpool.log" 2>&1
  echo "EXIT=$?" >> "$out/refpool.log"
  echo "$name refpool done $(date +%H:%M:%S)"
}

V4_EDIT_STRENGTH=0.15 V4_CFG=1.0 V4_NUM_STEPS="$steps" evalrun s0.15_cf1.0
V4_EDIT_STRENGTH=0.30 V4_CFG=1.5 V4_NUM_STEPS="$steps" evalrun s0.30_cf1.5
V4_IDENTITY=1 V4_EDIT_STRENGTH=0.0 V4_CFG=1.0 V4_NUM_STEPS="$steps" evalrun identity
V4_FORCE_NULL=1 V4_EDIT_STRENGTH=0.15 V4_CFG=1.0 V4_NUM_STEPS="$steps" evalrun nullcond_s0.15
echo REFPOOLDONE

