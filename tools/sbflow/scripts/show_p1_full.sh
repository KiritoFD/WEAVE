#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MIN = STY.index("Minimalism")
print(f"  {'arm':20s}{'lam':>5s}{'steps':>6s}{'dT':>9s}{'Top-1':>8s}{'ER':>7s}{'Imp':>7s}"
      f"{'Min':>7s}{'Roc':>7s}{'Uki':>7s}{'dM':>9s}{'DINO':>8s}{'LPIPS':>8s}{'sty':>5s}")
rows = [("P1_B0_ctrl218", 0, 800), ("P1_B1_distill218", 1, 800),
        ("P1_B2c_ctrl845", 0, 800), ("P1_B2_distill845", 1, 800),
        ("P1_B3_distill20", 20, 800), ("P1_B4_distill50", 50, 800),
        ("F_lam20", 20, 200)]
for a, lam, st in rows:
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        print(f"  {a:20s}  pending"); continue
    d = json.loads(p.read_text()); e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values())/len(pt) if pt else 0
    print(f"  {a:20s}{lam:5d}{st:6d}{e.get('mean_dT',0):+9.4f}{t1:8.3f}"
          f"{pt.get('Early_Renaissance',0):7.3f}{pt.get('Impressionism',0):7.3f}"
          f"{pt.get('Minimalism',0):7.3f}{pt.get('Rococo',0):7.3f}"
          f"{pt.get('Ukiyo_e',0):7.3f}{e.get('mean_dM',0):+9.4f}"
          f"{e.get('mean_dino_c',0):8.4f}{e.get('mean_lpips',0):8.4f}"
          f"{e.get('styles_improved')}/5")

print("\n  non-target Minimalism rate (from the confusion matrix, int-index safe)")
for a, _, _ in rows:
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        continue
    e = json.loads(p.read_text()).get("eval200") or {}
    conf = e.get("confusion_target_vs_pred")
    if not conf:
        print(f"  {a:20s}  no confusion"); continue
    h = sum(conf[i][MIN] for i in range(5) if i != MIN)
    t = sum(sum(conf[i]) for i in range(5) if i != MIN)
    print(f"  {a:20s} {h:3d}/{t:3d} = {h/max(1,t):.3f}")

print("\n  same50 dS")
for a, _, _ in rows:
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        continue
    e = json.loads(p.read_text()).get("eval200") or {}
    print(f"  {a:20s} same_dS={e.get('same50_mean_dS',0):+.4f} "
          f"ukiyo_dT={e.get('ukiyo_dT',0):+.4f}")
PY
