#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
import numpy as np
from pathlib import Path
from tools.v6_24h.stats import cluster_ci, paired_cluster_ci

STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MIN = STY.index("Minimalism")
ARMS = ["P1_B0_ctrl218", "P1_B3_distill20", "P1_B4_distill50"]

def load(a):
    p = Path(f"runs/v6_next/eval_750_{a}/eval.json")
    return json.loads(p.read_text()) if p.exists() else None

def split(rows):
    """eval_tiers does not write is_cross, so derive it: same-style iff target == source."""
    cross = [r for r in rows if r["tgt_style"] != r["src_style"]]
    same = [r for r in rows if r["tgt_style"] == r["src_style"]]
    return cross, same

store = {}
print("=" * 110)
print("750 PROTOCOL  (600 cross + 150 same)   split by target==source")
print("=" * 110)
print(f"  {'arm':20s}{'n_x':>5s}{'Top-1':>8s}{'CI':>22s}{'dT':>9s}{'margin':>9s}"
      f"{'DINO-C':>8s}{'LPIPS':>8s}   {'n_s':>4s}{'sameTop1':>9s}{'same dS':>9s}")
for a in ARMS:
    d = load(a)
    if d is None:
        print(f"  {a:20s}  missing"); continue
    cross, same = split(d["rows"])
    store[a] = cross
    t1 = [float(r["target_top1"]) for r in cross]
    ci = cluster_ci(t1, [r["src_stem"] for r in cross])
    print(f"  {a:20s}{len(cross):5d}{np.mean(t1):8.3f}"
          f"{'[%.3f,%.3f]' % (ci['ci'][0], ci['ci'][1]):>22s}"
          f"{np.mean([r['dT'] for r in cross]):+9.4f}"
          f"{np.mean([r['target_margin'] for r in cross]):+9.4f}"
          f"{np.mean([r['dino_c'] for r in cross]):8.4f}"
          f"{np.mean([r['lpips'] for r in cross]):8.4f}"
          f"   {len(same):4d}{np.mean([r['target_top1'] for r in same]):9.3f}"
          f"{np.mean([r['dS'] for r in same]):+9.4f}")

print("\n  per-target Top-1 (CROSS only)")
print(f"  {'arm':20s}" + "".join(f"{s[:9]:>11s}" for s in STY))
for a in ARMS:
    if a not in store:
        continue
    cells = []
    for s in STY:
        sub = [r for r in store[a] if r["tgt_style"] == s]
        cells.append(sum(r["target_top1"] for r in sub) / max(1, len(sub)))
    print(f"  {a:20s}" + "".join(f"{c:11.3f}" for c in cells))

print("\n  non-target Minimalism rate (cross only)")
for a in ARMS:
    if a not in store:
        continue
    conf = np.zeros((5, 5), dtype=int)
    for r in store[a]:
        conf[STY.index(r["tgt_style"])][int(r["pred_class"])] += 1
    h = sum(conf[i][MIN] for i in range(5) if i != MIN)
    t = sum(sum(conf[i]) for i in range(5) if i != MIN)
    print(f"  {a:20s} {h:3d}/{t:3d} = {h/max(1,t):.3f}")

print("\n" + "=" * 110)
print("PAIRED CLUSTERED DIFFERENCES vs P1_B0_ctrl218 (CROSS only, 150 source clusters)")
print("=" * 110)
base = store.get("P1_B0_ctrl218")
if base:
    cl = [r["src_stem"] for r in base]
    for a in ARMS[1:]:
        if a not in store:
            continue
        ra = store[a]
        dt = paired_cluster_ci([r["dT"] for r in ra], [r["dT"] for r in base], cl)
        t1 = paired_cluster_ci([float(r["target_top1"]) for r in ra],
                               [float(r["target_top1"]) for r in base], cl)
        print(f"  {a:20s} d(dT)  ={dt['diff']:+.4f} "
              f"CI[{dt['ci'][0]:+.4f},{dt['ci'][1]:+.4f}] excl0={dt['excludes_zero']}")
        print(f"  {'':20s} d(Top1)={t1['diff']:+.4f} "
              f"CI[{t1['ci'][0]:+.4f},{t1['ci'][1]:+.4f}] excl0={t1['excludes_zero']}")
PY
