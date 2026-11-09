#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
"""Full per-arm report. Reads each arm's report.json directly (state.json only carries a
subset of fields), so per-target and confusion data are always available."""
import json
from pathlib import Path

STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SH = {"Early_Renaissance": "ER", "Impressionism": "Imp", "Minimalism": "Min",
      "Rococo": "Roc", "Ukiyo_e": "Uki"}

state = json.loads(Path("runs/v6_24h/state.json").read_text())

records = []          # (name, status, report_dict_or_None)
a0p = Path("runs/v6/D1_c1_gentle/report.json")
if a0p.exists():
    records.append(("A0 (ref)", "ref", json.loads(a0p.read_text())))
for name, v in state.items():
    p = Path(f"runs/v6/{name}/report.json")
    records.append((name, v.get("status", "?"), json.loads(p.read_text()) if p.exists() else None))


def fmt(v, spec, dash="—"):
    return dash if v is None else format(v, spec)


print(f"{'arm':15s}{'status':9s}{'dT':>9s}{'dT>0':>6s}{'DINO-C':>8s}{'LPIPS':>8s}"
      f"{'sty+':>5s}{'sameDS':>8s}{'top1':>6s}{'min':>6s}")
print("-" * 80)
for name, status, r in records:
    e = (r or {}).get("eval200") or {}
    wall = state.get(name, {}).get("wall_sec")
    if not e.get("mean_dT"):
        print(f"{name:15s}{status:9s}{'—':>9s}{'—':>6s}{'—':>8s}{'—':>8s}{'—':>5s}"
              f"{'—':>8s}{'—':>6s}{(f'{wall/60:.1f}' if wall else '—'):>6s}")
        continue
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values()) / len(pt) if pt else None
    print(f"{name:15s}{status:9s}{e['mean_dT']:+9.4f}"
          f"{fmt(e.get('frac_dT_positive'), '.2f'):>6s}"
          f"{e['mean_dino_c']:8.4f}{e['mean_lpips']:8.4f}"
          f"{str(e.get('styles_improved')) + '/5':>5s}"
          f"{fmt(e.get('same50_mean_dS'), '+.4f'):>8s}"
          f"{fmt(t1, '.3f'):>6s}"
          f"{(f'{wall/60:.1f}' if wall else '—'):>6s}")

print()
print(f"{'per-target dT':15s}" + "".join(f"{SH[x]:>9s}" for x in STY))
for name, status, r in records:
    e = (r or {}).get("eval200") or {}
    pt = e.get("per_target_dT")
    if pt:
        print(f"{name:15s}" + "".join(f"{pt[x]:+9.4f}" for x in STY))

print()
print(f"{'per-target top1':15s}" + "".join(f"{SH[x]:>9s}" for x in STY))
for name, status, r in records:
    e = (r or {}).get("eval200") or {}
    pt = e.get("per_target_top1")
    if pt:
        print(f"{name:15s}" + "".join(f"{pt[x]:9.3f}" for x in STY))

print()
print("self-consistency (same-style reconstruction run THROUGH the network):")
print(f"{'arm':15s}{'sameDS':>9s}   note")
for name, status, r in records:
    e = (r or {}).get("eval200") or {}
    d = e.get("same50_mean_dS")
    if d is None:
        continue
    note = "self-consistent" if abs(d) < 0.02 else "NOT self-consistent"
    print(f"{name:15s}{d:+9.4f}   {note}")

# confusion matrix for the most recent completed arm that has one
for name, status, r in reversed(records):
    e = (r or {}).get("eval200") or {}
    conf = e.get("confusion_target_vs_pred")
    if conf:
        print()
        print(f"confusion (rows=target, cols=predicted CSD prototype) for {name}:")
        print(f"{'':10s}" + "".join(f"{SH[s]:>7s}" for s in STY))
        for i, s in enumerate(STY):
            print(f"{SH[s]:10s}" + "".join(f"{c:7d}" for c in conf[i]))
        break
PY
