#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
echo "=== tmux ==="
tmux list-panes -t suite40 -F '  #{pane_id} #{pane_pid} #{pane_current_command}' 2>/dev/null || echo "  no session"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== processes ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\.(suite40|train5|gen5|score_cpu|pretrain)" | grep -v grep | cut -c1-110
echo "=== suite state (after folding completed scores) ==="
python3 -u sbflow/fold_scores40.py >/dev/null 2>&1 || true
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/blog_sbflow/suite40/suite_state.json")
if not p.exists():
    print("  no state yet")
else:
    s = json.loads(p.read_text())
    for k in sorted(s):
        v = s[k]
        e = v.get("eval") or {}
        if "target_top1" in e:
            print(f"  {k} {v['status']:18s} top1={e['target_top1']:.4f} "
                  f"margin={e['target_margin']:+.4f} dT={e['csd_dT']:+.4f} "
                  f"train={v.get('train_seconds',0):.0f}s")
        else:
            print(f"  {k} {v['status']:18s} {v.get('error','')[:70]}")
PY
echo "=== recent log ==="
tail -n 10 "$ROOT/logs/suite.log" 2>/dev/null || echo "  (none)"
