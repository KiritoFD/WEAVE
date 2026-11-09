#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== step=50 speed line (exp01) ==="
grep -a "step=50/" runs/blog_sbflow/suite40/logs/01_train.log | head -2 || echo "  missing"
echo "=== state timing fields ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/suite_state.json")
s=json.loads(p.read_text())
for k in sorted(s):
 v=s[k]
 print(f"  {k}: status={v.get('status')} "
       f"train={v.get('train_seconds')} s/upd={v.get('sec_per_update')} "
       f"first50={v.get('first50_sec_per_update')} peak={v.get('peak_vram_mib')} "
       f"exposures={v.get('exposures')}")
PY
