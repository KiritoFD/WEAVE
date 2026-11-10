#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== task files ==="
for f in eval_protocol/tasks_750.json runs/v6_24h/heldout400_tasks.json \
         runs/v6_24h/same150_tasks.json runs/v6_24h/screen50_tasks.json; do
  if [ -f "$f" ]; then
    python3 - "$f" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
if isinstance(d, dict):
    t = d.get("tasks") or []
    proto = d.get("protocol") or {}
    print(f"  {p.name:32s} dict keys={list(d.keys())[:5]} n_tasks={len(t)} protocol={proto.get('name')}")
    if t:
        k = t[0]
        print(f"      task keys: {list(k.keys())}")
        print(f"      sample: {k}")
else:
    print(f"  {p.name:32s} list n={len(d)}")
    print(f"      sample: {d[0]}")
PY
  else
    echo "  $f  MISSING"
  fi
done
