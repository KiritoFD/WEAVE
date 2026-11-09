#!/usr/bin/env bash
# Inventory the V6 ablations that already exist on the GPU box.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== runs/v6 tree (depth 2) ==="
find runs/v6 -maxdepth 2 -type f | head -40
echo
echo "=== report.json keys per variant ==="
python3 - <<'PY'
import json
from pathlib import Path
for d in sorted(Path("runs/v6").iterdir()):
    if not d.is_dir():
        continue
    rp = d / "report.json"
    files = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    if rp.exists():
        try:
            r = json.loads(rp.read_text())
            keys = list(r.keys())
            print(f"  {d.name}: report keys={keys[:10]}")
            for k in ("config", "eval", "metrics", "summary"):
                if k in r and isinstance(r[k], dict):
                    print(f"      {k}: {list(r[k].keys())[:12]}")
        except Exception as e:
            print(f"  {d.name}: report unreadable ({e})")
    else:
        print(f"  {d.name}: no report.json, files={files[:6]}")
PY
