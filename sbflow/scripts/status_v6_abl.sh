#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== tmux ==="
tmux ls 2>&1 | head -6
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== procs ==="
ps -eo pid,etime,args --no-headers | grep -E "train_eval_v6" | grep -v grep | cut -c1-110 || echo "  none"
echo "=== ablation logs ==="
for n in A1_no_content A2_no_style A3_canon_none A4_dim24 A5_dim32 A6_dim40 A7_steps400; do
  f="runs/v6/logs/${n}.log"
  if [ -f "$f" ]; then
    echo "  -- $n"
    grep -aE "params=|Eval200|Error|error" "$f" | tail -2 | sed 's/^/     /'
  fi
done
