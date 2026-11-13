#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== git rev-parse direct ==="
git rev-parse --short HEAD 2>&1 || true
echo "=== git status ==="
git status -sb 2>&1 | head -3 || true
echo "=== state git_sha values ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/suite_state.json")
if p.exists():
 s=json.loads(p.read_text())
 for k in sorted(s): print(f"  {k}: git_sha={s[k].get('git_sha')!r}")
PY
echo "=== config has vae/seed ==="
python3 - <<'PY'
import json
d=json.load(open("configs/sbflow/exp01.json"))
print("  vae key present:", "vae" in d, " seed:", d.get("seed"))
PY
