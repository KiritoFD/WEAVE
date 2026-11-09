#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
echo "=== exp01 checkpoints ==="
ls -lh "$ROOT/exp01"/*.pt 2>/dev/null || echo "  none"
echo "=== exp01 config resolved ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/configs/exp01.json")
if p.exists():
 d=json.loads(p.read_text())
 for k in ("steps","batch","equiv_batch","save_equiv","full_equiv","lr_cycle_steps","save_interval","grad_microbatch","mixed_batch","equal_weight_stats"):
  print(f"  {k}={d.get(k)}")
PY
echo "=== fork config resolved paths ==="
for i in 24 27 32; do
 python3 - <<PY
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/configs/exp$i.json")
if p.exists():
 d=json.loads(p.read_text())
 f=Path(d["resume"])
 print(f"  exp$i resume={f} exists={f.exists()}")
PY
done
echo "=== exp01 train log tail ==="
tail -n 6 "$ROOT/logs/01_train.log" 2>/dev/null || true
