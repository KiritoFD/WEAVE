#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== fix present? ==="
grep -n 'src_stem": t\["src_stem"\]' tools/v6/train_eval_v6.py | head -2
echo "=== running arm ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -1 | cut -c1-120
echo "=== queue state ==="
python3 - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("runs/v6_24h/state.json").read_text())
for k, v in s.items():
    print(f"  {k:16s} {v.get('status')}")
PY
