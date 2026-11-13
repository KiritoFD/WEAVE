#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
echo "=== exp01 score artifacts ==="
ls -lh "$ROOT/score_01_quick200/" 2>/dev/null || echo "  missing score dir"
echo "=== exp01 score log tail ==="
tail -n 8 "$ROOT/logs/01_score.log" 2>/dev/null || echo "  no log"
echo "=== exp01 gen manifest ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/gen_01_quick200/manifest.json")
if p.exists():
 d=json.loads(p.read_text()); print("  tasks", len(d["tasks"]), "images", sum(1 for t in d["tasks"] if t.get("image")))
else: print("  missing manifest")
PY
echo "=== all python processes ==="
ps -eo pid,etime,args --no-headers | grep -E "python3 -m sbflow|/usr/bin/python3 -m sbflow" | grep -v grep | cut -c1-120 || echo "  none"
