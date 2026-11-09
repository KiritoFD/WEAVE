#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== tmux ==="
tmux ls 2>&1 | head -6
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== queue procs ==="
ps -eo pid,etime,args --no-headers | grep -E "v6_24h.queue|train_eval_v6" | grep -v grep | cut -c1-100 || echo "  none"
echo "=== state ==="
if [ -f runs/v6_24h/state.json ]; then
  python3 - <<'PY'
import json
s = json.load(open("runs/v6_24h/state.json"))
print(f"{'arm':16s} {'status':11s} {'dT':>9s} {'dino':>7s} {'lpips':>7s} {'min':>6s}")
for k, v in s.items():
    e = v.get("eval200") or {}
    f = lambda x, p="": (f"{x:{p}}" if x is not None else "—")
    print(f"{k:16s} {v.get('status','?'):11s} "
          f"{f(e.get('mean_dT'),'+.4f'):>9s} {f(e.get('mean_dino_c'),'.4f'):>7s} "
          f"{f(e.get('mean_lpips'),'.4f'):>7s} "
          f"{(v.get('wall_sec') or 0)/60:6.1f}")
PY
else
  echo "  no state.json yet"
fi
echo "=== queue log tail ==="
tail -n 6 runs/v6_24h/queue.log 2>/dev/null || echo "  no log"
