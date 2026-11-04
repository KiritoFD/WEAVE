#!/usr/bin/env bash
# 逐个 checkpoint 跑 v4.evaluate（去噪分桶 / 风格探针 / 纯生成 / swap / 记忆检查 + 解码图）。
# Usage: v4_eval_ckpts.sh <run_dir> [steps] [gen_per_style]
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_dir="${1:-runs/v4/prior}"; steps="${2:-20}"; n="${3:-16}"
[ -d "$run_dir" ] || exit 1
exec >> "$run_dir/eval_ckpts.log" 2>&1
for f in $(ls "$run_dir"/step_*.pt "$run_dir"/best.pt 2>/dev/null); do
  name=$(basename "$f" .pt)
  out="$run_dir/eval_$name"
  [ -f "$out/eval.json" ] && continue
  mkdir -p "$out"
  python3 -m v4.evaluate --ckpt "$f" --run-dir "$run_dir" --out "$out" \
    --gen-per-style "$n" --steps "$steps" --decode > "$out/eval.log" 2>&1
  echo "$name EXIT=$? $(date +%H:%M:%S)"
done
