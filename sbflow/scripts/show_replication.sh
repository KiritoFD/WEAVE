#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== Rococo replication across seeds (the key question) ==="
python3 - <<'PY'
import json
from pathlib import Path
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

def get(a):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values()) / len(pt) if pt else 0.0
    return {"seed": d.get("seed"), "dT": e.get("mean_dT"), "t1": t1,
            "dM": e.get("mean_dM"), "dino": e.get("mean_dino_c"),
            "lpips": e.get("mean_lpips"), "pt": pt,
            "same": e.get("same50_mean_dS"), "sty": e.get("styles_improved")}

pairs = [("42", "P2_G0_bf16", "P2_G1b_bf16"), ("43", "Q1b_G0_s43", "Q1b_G1b_s43")]
print(f"  {'seed':>5s}{'arm':>6s}{'dT':>9s}{'Top-1':>8s}{'Rococo':>9s}{'ER':>7s}"
      f"{'Imp':>7s}{'Min':>7s}{'Uki':>7s}{'DINO':>8s}")
for seed, g0, g1 in pairs:
    for tag, name in (("G0", g0), ("G1b", g1)):
        r = get(name)
        if r is None:
            print(f"  {seed:>5s}{tag:>6s}   pending"); continue
        print(f"  {seed:>5s}{tag:>6s}{r['dT']:+9.4f}{r['t1']:8.3f}{r['pt']['Rococo']:9.3f}"
              f"{r['pt']['Early_Renaissance']:7.3f}{r['pt']['Impressionism']:7.3f}"
              f"{r['pt']['Minimalism']:7.3f}{r['pt']['Ukiyo_e']:7.3f}{r['dino']:8.4f}")
    a, b = get(g0), get(g1)
    if a and b:
        print(f"  {seed:>5s}{'delta':>6s}{b['dT']-a['dT']:+9.4f}{b['t1']-a['t1']:8.3f}"
              f"{b['pt']['Rococo']-a['pt']['Rococo']:+9.3f}"
              f"{'':7s}{'':7s}{'':7s}{'':7s}{b['dino']-a['dino']:+8.4f}")
    print()
PY
