#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== any sbflow/v6 process ==="
ps -eo pid,args --no-headers | grep -E "train_eval_v6|sbflow\." | grep -v grep | cut -c1-90 || echo "  none"
echo "=== tmux ==="
tmux ls 2>&1 || true
echo "=== v6 variants with report.json ==="
for d in runs/v6/*/; do
  n=$(basename "$d")
  if [ -f "$d/report.json" ]; then
    has=$(python3 -c "
import json,sys
r=json.load(open('$d/report.json'))
print('eval200' if r.get('eval200') else 'no-eval200')
" 2>/dev/null)
    echo "  $n : $has"
  fi
done
echo "=== A-arm logs present (partial runs) ==="
ls -1 runs/v6/logs/ 2>/dev/null | head -20
