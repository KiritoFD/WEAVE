#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
echo "=== exp27 train log (first 50 + tail) ==="
grep -a "step=50/" "$ROOT/logs/27_train.log" 2>/dev/null | head -1 || true
tail -n 6 "$ROOT/logs/27_train.log" 2>/dev/null || true
echo "=== exp27 resolved config ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/configs/exp27.json")
if p.exists():
 d=json.loads(p.read_text())
 for k in ("steps","batch","grad_microbatch","online","online_steps","lr_cycle_steps","resume"):
  print(f"  {k}={d.get(k)}")
PY
echo "=== estimated time for remaining E/F arms ==="
python3 - <<'PY'
import json
from pathlib import Path
cfg=Path("configs/sbflow")
heavy=[]
for i in list(range(27,34))+[36,37,39]:
 d=json.loads((cfg/f"exp{i:02d}.json").read_text())
 heavy.append((f"{i:02d}",d.get("online","none"),d.get("grad_microbatch",0),d.get("steps")))
print(" ",heavy)
PY
