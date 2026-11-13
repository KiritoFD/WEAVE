#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/v11
echo "=== tmux ==="
tmux ls 2>/dev/null | head -6
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== single-GPU check ==="
bash -u sbflow/scripts/check_single_gpu.sh 2>&1 | tail -4
echo "=== p1 log ==="
tail -n 10 "$ROOT/logs/p1.log" 2>/dev/null || echo "  no p1 log"
echo "=== arm train logs ==="
for a in d1 d2 e1 e2; do
  if [ -f "$ROOT/logs/${a}_train.log" ]; then
    echo "  -- $a"; tail -n 3 "$ROOT/logs/${a}_train.log"
  fi
done
echo "=== p1 state ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/blog_sbflow/v11/p1_state.json")
if not p.exists():
    print("  no state yet")
else:
    s = json.loads(p.read_text())
    for k in sorted(s):
        v = s[k]
        e = v.get("eval") or {}
        extra = (f"top1={e['target_top1']:.4f} dT={e['csd_dT']:+.4f}"
                 if "target_top1" in e else v.get("error", "")[:60])
        print(f"  {k}: {v.get('status')} {extra}")
PY
