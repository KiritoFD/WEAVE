#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("runs/v6_24h/state.json").read_text())
# reference A0 from the existing report
a0 = json.loads(Path("runs/v6/D1_c1_gentle/report.json").read_text())["eval200"]

STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SH = {"Early_Renaissance": "ER", "Impressionism": "Imp", "Minimalism": "Min",
      "Rococo": "Roc", "Ukiyo_e": "Uki"}

print(f"{'arm':14s} {'status':10s} {'dT':>9s} {'dino':>7s} {'lpips':>7s} {'sty+':>5s} {'min':>6s}")
print(f"{'A0 (ref)':14s} {'-':10s} {a0['mean_dT']:+9.4f} {a0['mean_dino_c']:7.4f} "
      f"{a0['mean_lpips']:7.4f} {str(a0['styles_improved'])+'/5':>5s} {'-':>6s}")
for k, v in s.items():
    e = v.get("eval200") or {}
    if not e:
        print(f"{k:14s} {v.get('status','?'):10s} {'-':>9s} {'-':>7s} {'-':>7s} {'-':>5s} "
              f"{(v.get('wall_sec') or 0)/60:6.1f}")
        continue
    print(f"{k:14s} {v.get('status','?'):10s} {e['mean_dT']:+9.4f} "
          f"{e['mean_dino_c']:7.4f} {e['mean_lpips']:7.4f} "
          f"{str(e.get('styles_improved'))+'/5':>5s} {(v.get('wall_sec') or 0)/60:6.1f}")

print()
print(f"{'per-target dT':14s}" + "".join(f"{SH[x]:>9s}" for x in STY))
def row(name, pt):
    return f"{name:14s}" + "".join(f"{pt.get(x, float('nan')):+9.4f}" for x in STY)
print(row("A0 (ref)", a0["per_target_dT"]))
for k, v in s.items():
    e = v.get("eval200") or {}
    if e:
        print(row(k, v.get("per_target_dT") or {}))
PY
