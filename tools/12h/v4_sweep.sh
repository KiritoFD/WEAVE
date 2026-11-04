#!/usr/bin/env bash
# V4 SDEdit 小集扫参：每风格 8 张内容图 -> 40 对；协议其余部分完全沿用 paper/512/TRT。
# Usage: v4_sweep.sh <ckpt> <root_out> [steps=20] [seed=0] [max_src=8]
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ckpt="$1"; root="$2"; steps="${3:-20}"; seed="${4:-0}"; maxsrc="${5:-8}"
mkdir -p "$root"
exec >> "$root/sweep.log" 2>&1

evalrun() {
  name="$1"; out="$root/$name"; mkdir -p "$out"
  python3 tools/12h/eval_fast_trt_fp16.py --protocol paper --checkpoint "$ckpt" --output "$out" \
    --device cuda --batch 8 --metric-batch 32 --decode-batch 8 --image-size 512 --workers 4 \
    --no-save-images --gen-dtype fp32 --vae-dtype fp32 --sync-timing --decoder trt \
    --max-src-samples "$maxsrc" --num-steps "$steps" --seed "$seed" --backbone "v4_$name" \
    > "$out/eval.log" 2>&1
  echo "EXIT=$?" >> "$out/eval.log"
  echo "$name done $(date +%H:%M:%S)"
}

for s in 0.15 0.30 0.45 0.60; do
  for c in 1.0 1.5 2.0 3.0; do
    V4_NUM_STEPS="$steps" V4_EDIT_STRENGTH="$s" V4_CFG="$c" V4_SEED="$seed" \
      evalrun "s${s}_cf${c}"
  done
done
V4_IDENTITY=1 evalrun identity
V4_FORCE_NULL=1 V4_EDIT_STRENGTH=0.45 V4_CFG=1.0 V4_NUM_STEPS="$steps" \
  evalrun nullcond_s0.45_cf1.0
V4_EDIT_STRENGTH=0.45 V4_CFG=1.0 V4_NUM_STEPS="$steps" V4_SEED="$seed" \
  evalrun repeat_s0.45_cf1.0
