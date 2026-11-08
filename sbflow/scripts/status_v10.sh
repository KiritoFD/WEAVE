#!/usr/bin/env bash
# Inspect the detached V10 suite: tmux session state, progress counters, recent log lines.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=${1:-v10}
ROOT=runs/blog_sbflow/v10

echo "=== tmux ==="
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux list-panes -t "$SESSION" -F '  pane #{pane_id} pid=#{pane_pid} #{pane_current_command}'
else
  echo "  session '$SESSION' NOT running"
fi

echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo "=== progress ==="
if [ -f "$ROOT/suite_state.json" ]; then
  python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/blog_sbflow/v10/suite_state.json")
s = json.loads(p.read_text())
from collections import Counter
c = Counter(v.get("status") for v in s.values())
print("  status:", dict(c), f"({len(s)} arms seen)")
for k in sorted(s):
    v = s[k]
    e = v.get("eval") or {}
    if "target_top1" in e:
        print(f"  {k} {v['status']:22s} top1={e['target_top1']:.4f} "
              f"margin={e['target_margin']:+.4f} dT={e['csd_dT']:+.4f} "
              f"swd={e.get('csd_swd', float('nan')):.4f} "
              f"train={v.get('train_seconds', 0):.0f}s")
    else:
        print(f"  {k} {v['status']}")
PY
else
  echo "  no suite_state.json yet"
fi

echo "=== recent log lines ==="
tail -n 8 "$ROOT/logs/gate.log" 2>/dev/null || echo "  (no gate log)"
