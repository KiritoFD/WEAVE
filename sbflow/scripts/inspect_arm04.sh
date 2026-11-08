#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== exp_04 contents ==="
ls -lh runs/blog_sbflow/v10/exp_04/ 2>/dev/null || echo "  none"
echo "=== checkpoint metadata ==="
python3 - <<'PY'
import glob, torch
for f in sorted(glob.glob("runs/blog_sbflow/v10/exp_04/model_step*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    print(f"  {f}")
    print(f"    step={d.get('step')} styles={d.get('styles')}")
    print(f"    model_kwargs={d.get('model_kwargs')}")
    print(f"    ae={d.get('ae')} sf={d.get('ae_scaling_factor')}")
    print(f"    config.batch={d.get('config',{}).get('batch')} patch={d.get('config',{}).get('patch')}")
PY
echo "=== running python ==="
ps -eo pid,etime,args | grep -E "sbflow\." | grep -v grep | head -5
echo "=== v10 dirs ==="
ls -d runs/blog_sbflow/v10/*/ 2>/dev/null
