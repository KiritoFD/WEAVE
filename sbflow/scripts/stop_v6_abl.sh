#!/usr/bin/env bash
# Stop the V6 ablation grid (wrong base_dim run) and confirm the GPU is free.
set -uo pipefail
if tmux has-session -t v6abl 2>/dev/null; then
  tmux kill-session -t v6abl && echo "  killed tmux v6abl"
fi
pkill -f "train_eval_v6" 2>/dev/null && echo "  killed train_eval_v6" || echo "  no train_eval_v6"
sleep 4
echo "=== tmux ==="
tmux ls 2>&1 || true
echo "=== v6 procs ==="
ps -eo pid,args --no-headers | grep -E "train_eval_v6|score_diff" | grep -v grep | cut -c1-100 || echo "  none"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
