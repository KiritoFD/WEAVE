#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== P1 queue ==="
grep -aE "STEP |OK rc=|FAILED|FATAL|DONE|GPU free|distillation" runs/v6_24h/queue_p1.log 2>/dev/null | tail -n 12 || echo "  no log"
echo
echo "=== running ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -1 | cut -c1-64
echo
echo "=== P1 arms ==="
python3 - <<'PY'
import json
from pathlib import Path
arms = ("P1_B0_ctrl218", "P1_B1_distill218", "P1_B2c_ctrl845", "P1_B2_distill845")
print(f"  {'arm':20s}{'dT':>9s}{'Top-1':>8s}{'Roc':>7s}{'Uki':>7s}{'dM':>9s}"
      f"{'DINO':>8s}{'LPIPS':>8s}{'sty':>5s}")
for a in arms:
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        print(f"  {a:20s} pending"); continue
    d = json.loads(p.read_text()); e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values())/len(pt) if pt else 0
    print(f"  {a:20s}{e.get('mean_dT',0):+9.4f}{t1:8.3f}{pt.get('Rococo',0):7.3f}"
          f"{pt.get('Ukiyo_e',0):7.3f}{e.get('mean_dM',0):+9.4f}"
          f"{e.get('mean_dino_c',0):8.4f}{e.get('mean_lpips',0):8.4f}"
          f"{e.get('styles_improved')}/5")
PY
echo
echo "=== guards ==="
python3 - <<'PY'
import torch
from pathlib import Path
def m(a):
    p = Path(f"runs/v6/{a}/{a}.pt")
    return torch.load(p, map_location="cpu", weights_only=False)["model"] if p.exists() else None
def d(x, y):
    return max(float((x[k].float()-y[k].float()).abs().max()) for k in x)
for c, t in (("P1_B0_ctrl218", "P1_B1_distill218"),
             ("P1_B2c_ctrl845", "P1_B2_distill845")):
    a, b = m(c), m(t)
    if a is None or b is None:
        print(f"  {t}: pending"); continue
    v = d(a, b)
    print(f"  {t} vs {c}: max|dW|={v:.3e} -> "
          + ("FATAL: distill did nothing" if v == 0 else "reached the weights"))
PY
echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
