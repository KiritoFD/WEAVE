#!/usr/bin/env bash
set -uo pipefail
pkill -f "train_eval_v6.py --name G1b" 2>/dev/null || true
sleep 3
if ps -eo args --no-headers | grep train_eval_v6 | grep -v grep >/dev/null 2>&1; then
  echo "still running:"
  ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | cut -c1-70
else
  echo "KILLED - no train_eval_v6 process"
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
