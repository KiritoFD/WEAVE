#!/usr/bin/env bash
# V4 训练启动器（供 tmux 调用）。参数：run_name [microbatch] [grad_accum] [max_steps]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run="${1:-prior}"; mb="${2:-16}"; accum="${3:-2}"; steps="${4:-0}"
if [ "$steps" -gt 0 ]; then
  python3 -u -m v4.train --run-name "$run" --batch-size "$mb" --grad-accum "$accum" --max-steps "$steps"
else
  python3 -u -m v4.train --run-name "$run" --batch-size "$mb" --grad-accum "$accum"
fi
