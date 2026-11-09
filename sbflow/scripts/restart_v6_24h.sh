#!/usr/bin/env bash
# Stop the queue and the in-flight arm so the next start uses the fixed code.
# Done arms are skipped automatically on restart; N5/N4 are retried in priority order.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
if tmux has-session -t v6_24h 2>/dev/null; then
  tmux kill-session -t v6_24h && echo "  killed tmux v6_24h"
fi
pkill -f "tools.v6_24h.queue" 2>/dev/null && echo "  killed queue" || true
pkill -f "train_eval_v6" 2>/dev/null && echo "  killed train_eval_v6" || true
sleep 4
rm -f runs/v6_24h/.gpu.lock
echo "=== remaining ==="
ps -eo pid,args --no-headers | grep -E "v6_24h.queue|train_eval_v6" | grep -v grep || echo "  none"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== state ==="
python3 - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("runs/v6_24h/state.json").read_text())
for k, v in s.items():
    print(f"  {k:16s} {v.get('status')}")
PY
