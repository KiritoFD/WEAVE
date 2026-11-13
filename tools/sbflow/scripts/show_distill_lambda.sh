#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
print(f"  {'arm':16s}{'lam':>6s}{'steps':>7s}{'dT':>9s}{'Top-1':>8s}{'Roc':>7s}{'Uki':>7s}"
      f"{'dM':>9s}{'DINO':>8s}{'LPIPS':>8s}{'sty':>5s}")
rows = [("P1_B0_ctrl218", 0, 800), ("P1_B1_distill218", 1, 800),
        ("P1_B2c_ctrl845", 0, 800), ("P1_B2_distill845", 1, 800),
        ("F_lam1", 1, 200), ("F_lam5", 5, 200), ("F_lam20", 20, 200)]
for a, lam, st in rows:
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        print(f"  {a:16s}  pending"); continue
    d = json.loads(p.read_text()); e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values())/len(pt) if pt else 0
    print(f"  {a:16s}{lam:6d}{st:7d}{e.get('mean_dT',0):+9.4f}{t1:8.3f}"
          f"{pt.get('Rococo',0):7.3f}{pt.get('Ukiyo_e',0):7.3f}"
          f"{e.get('mean_dM',0):+9.4f}{e.get('mean_dino_c',0):8.4f}"
          f"{e.get('mean_lpips',0):8.4f}{e.get('styles_improved')}/5")
PY
