#!/usr/bin/env bash
# Stop the V11 direction: kill its tmux sessions and any running sbflow jobs.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
for s in v11p3long v11chain v11p1 v11p3; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s" && echo "  killed session $s"
  fi
done
pkill -f "sbflow.traj_check" 2>/dev/null && echo "  killed traj_check" || true
pkill -f "sbflow.train5" 2>/dev/null && echo "  killed train5" || true
pkill -f "sbflow.gen5" 2>/dev/null && echo "  killed gen5" || true
pkill -f "sbflow.score_cpu" 2>/dev/null && echo "  killed score_cpu" || true
pkill -f "sbflow.gen_uncond" 2>/dev/null && echo "  killed gen_uncond" || true
sleep 3
echo "=== remaining sessions ==="
tmux ls 2>&1 || true
echo "=== remaining sbflow procs ==="
ps -eo pid,args --no-headers | grep "sbflow\." | grep -v grep | cut -c1-100 || echo "  none"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
