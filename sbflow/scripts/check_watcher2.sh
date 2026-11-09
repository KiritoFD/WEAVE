#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== watcher processes ==="
ps -eo pid,etime,args --no-headers | grep -E "continue_suite40|suite40 .*--arms rest" | grep -v grep | cut -c1-130 || echo "  none"
echo "=== watcher pane ==="
tmux capture-pane -p -t suite40_continue 2>/dev/null | tail -8 || true
echo "=== core state ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/suite_state.json")
s=json.loads(p.read_text()) if p.exists() else {}
print(" ",{k:v.get("status") for k,v in sorted(s.items())})
PY
