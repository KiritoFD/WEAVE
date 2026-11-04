#!/usr/bin/env bash
# 在 tmux 里跑训练并落盘日志：v4_train_logged.sh <run> [mb] [accum] [max_steps]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
run="${1:-prior}"
bash tools/12h/v4_train.sh "$@" > "runs/v4/${run}_train.log" 2>&1
