#!/usr/bin/env bash
# V4 评测入口：沿用 eval_path_trt.sh 的固定协议（512 / CLIP-B32 / DINOv2-s / 750 all-pairs / b8 / TRT fp16），
# 只把 V4 的推理超参通过环境变量注入。NFE 与旧模型 8 步 ODE 不同，报告里必须分开写。
# Usage: eval_v4_trt.sh <tag> <ckpt> <out_dir> [batch=8] [steps=20] [strength=0.45] [cfg=1.0] [seed=0]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
tag="$1"; ckpt="$2"; out="$3"
batch="${4:-8}"; steps="${5:-20}"; strength="${6:-0.45}"; cfg="${7:-1.0}"; seed="${8:-0}"
export V4_NUM_STEPS="$steps"
export V4_EDIT_STRENGTH="$strength"
export V4_CFG="$cfg"
export V4_SEED="$seed"
mkdir -p "$out"
{ echo "ckpt=$ckpt batch=$batch steps=$steps strength=$strength cfg=$cfg seed=$seed"; } > "$out/v4_run_meta.txt"
bash tools/12h/eval_path_trt.sh "$tag" "$ckpt" "$out" "$batch"
