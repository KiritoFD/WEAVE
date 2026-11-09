#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
"""How much does the transformation actually move the image in CSD space?

dT = cos(gen, p_tgt) - cos(src, p_tgt)   gain toward the target prototype
dS = cos(gen, p_src) - cos(src, p_src)   loss of the SOURCE prototype
dM = dT - dS                              net margin change

A near-identity operator has dT ~ 0, dS ~ 0. An operator that genuinely changes style
identity should show dS clearly negative (it stops looking like the source) and dT clearly
positive. The ratio |dT| / |dS| says whether the movement is "toward target" or "away from
source", and the absolute size says how far it went at all.
"""
import json
from pathlib import Path

state = json.loads(Path("runs/v6_24h/state.json").read_text())
rows = []
a0p = Path("runs/v6/D1_c1_gentle/report.json")
if a0p.exists():
    rows.append(("A0 (ref)", json.loads(a0p.read_text())["eval200"]))
for k, v in state.items():
    p = Path(f"runs/v6/{k}/report.json")
    if p.exists():
        rows.append((k, json.loads(p.read_text()).get("eval200") or {}))

print(f"{'arm':16s}{'dT':>9s}{'dS':>9s}{'dM':>9s}{'|dT|/|dS|':>11s}   reading")
print("-" * 84)
for name, e in rows:
    dT, dS = e.get("mean_dT"), e.get("mean_dS")
    if dT is None or dS is None:
        continue
    dM = e.get("mean_dM")
    ratio = abs(dT) / abs(dS) if dS else float("inf")
    if abs(dS) < 0.02:
        note = "source identity almost untouched"
    elif abs(dS) < 0.06:
        note = "source identity weakened slightly"
    else:
        note = "source identity substantially weakened"
    print(f"{name:16s}{dT:+9.4f}{dS:+9.4f}{(dM if dM is not None else dT - dS):+9.4f}"
          f"{ratio:11.2f}   {note}")

print()
print("Interpretation: if dS is near zero the operator leaves the source identity intact and")
print("only nudges toward the target, which is exactly what 'pred == source 91% of the time'")
print("implies. A genuine style transfer would show dS clearly negative.")
PY
