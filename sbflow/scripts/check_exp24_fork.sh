#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
echo "=== exp24 resolved config ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/configs/exp24.json")
if not p.exists(): print("  no resolved config yet")
else:
 d=json.loads(p.read_text())
 for k in ("experiment_id","steps","batch","resume","lr_cycle_steps","dist_loss","style_enc","style_enc_ckpt","ode_steps_train","grad_microbatch"):
  print(f"  {k}={d.get(k)}")
PY
echo "=== fork checkpoint ==="
python3 - <<'PY'
import json, torch
from pathlib import Path
p=Path("runs/blog_sbflow/suite40/configs/exp24.json")
if p.exists():
 f=Path(json.loads(p.read_text()).get("resume",""))
 print(f"  path={f} exists={f.exists()}")
 if f.exists():
  d=torch.load(f,map_location="cpu",weights_only=False)
  print(f"  step={d.get('step')} equiv_step={d.get('equiv_step')} "
        f"has_model={('model' in d)} has_optimizer={('optimizer' in d)} "
        f"has_scheduler={('scheduler' in d)} has_rng={('rng' in d)}")
PY
echo "=== exp24 train log head ==="
head -n 4 "$ROOT/logs/24_train.log" 2>/dev/null || true
