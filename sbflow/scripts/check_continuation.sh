#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== tmux sessions ==="
tmux ls 2>&1 | head -10 || true
echo "=== relevant processes ==="
ps -eo pid,etime,args --no-headers | grep -E "continue_suite40|suite40|sbflow.train5|sbflow.gen5|sbflow.score_cpu" | grep -v grep | cut -c1-130 || echo "  none"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== continuation log ==="
tail -n 5 runs/blog_sbflow/suite40/logs/continuation.log 2>/dev/null || echo "  (none)"
echo "=== suite state ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/suite_state.json")
if not p.exists(): print("  no state")
else:
 s=json.loads(p.read_text())
 for k in sorted(s): print(f"  {k} {s[k].get('status')}")
PY
