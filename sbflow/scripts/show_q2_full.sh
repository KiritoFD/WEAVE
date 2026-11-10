#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import json
import numpy as np
from pathlib import Path
from tools.v6_24h.stats import paired_cluster_ci

STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
BASE = "P2_G0_bf16"

def load(a):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    e = d.get("eval200") or {}
    return {"e": e, "rows": e.get("rows") or [], "seed": d.get("seed"),
            "rk": d.get("rank_lambda"), "emb": d.get("embed_style_lambda"),
            "tr": d.get("trust_lambda")}

def t1(e):
    pt = e.get("per_target_top1") or {}
    return sum(pt.values())/len(pt) if pt else 0.0

arms = ["P2_G0_bf16", "P2_G1a_bf16", "P2_G1b_bf16", "Q2a_embed_style",
        "Q2b_trust", "Q2c_embed_trust"]
D = {a: load(a) for a in arms}
D = {k: v for k, v in D.items() if v}

print("=" * 104)
print("Q2 SUMMARY  (Val200, 200 cross-style tasks)")
print("=" * 104)
print(f"  {'arm':20s}{'rk':>7s}{'emb':>7s}{'trust':>7s}{'dT':>9s}{'Top-1':>8s}"
      f"{'margin':>9s}{'DINO':>8s}{'LPIPS':>8s}{'sty':>5s}")
for a in arms:
    if a not in D:
        continue
    e = D[a]["e"]
    print(f"  {a:20s}{D[a]['rk'] or 0:7.3f}{D[a]['emb'] or 0:7.2f}{D[a]['tr'] or 0:7.2f}"
          f"{e['mean_dT']:+9.4f}{t1(e):8.3f}{e['mean_dM']:+9.4f}"
          f"{e['mean_dino_c']:8.4f}{e['mean_lpips']:8.4f}{e['styles_improved']}/5")

print("\n" + "=" * 104)
print(f"PAIRED CLUSTERED DIFFERENCES vs {BASE}")
print("=" * 104)
r0 = D[BASE]["rows"]
c0 = [r["src_stem"] for r in r0]
for a in arms[1:]:
    if a not in D:
        continue
    ra = D[a]["rows"]
    d_dt = paired_cluster_ci([r["dT"] for r in ra], [r["dT"] for r in r0], c0)
    d_t1 = paired_cluster_ci([float(r["target_top1"]) for r in ra],
                             [float(r["target_top1"]) for r in r0], c0)
    print(f"  {a:20s} d(dT)={d_dt['diff']:+.4f} CI[{d_dt['ci'][0]:+.4f},{d_dt['ci'][1]:+.4f}]"
          f" excl0={d_dt['excludes_zero']}")
    print(f"  {'':20s} d(Top1)={d_t1['diff']:+.4f} "
          f"CI[{d_t1['ci'][0]:+.4f},{d_t1['ci'][1]:+.4f}] excl0={d_t1['excludes_zero']}")

print("\n" + "=" * 104)
print("PER-TARGET TOP-1")
print("=" * 104)
print(f"  {'arm':20s}" + "".join(f"{s[:9]:>11s}" for s in STY))
for a in arms:
    if a not in D:
        continue
    pt = D[a]["e"].get("per_target_top1") or {}
    print(f"  {a:20s}" + "".join(f"{pt.get(s,0):11.3f}" for s in STY))

print("\n" + "=" * 104)
print("NON-TARGET MINIMALISM MISCLASSIFICATION (absolute rate)")
print("=" * 104)
for a in arms:
    if a not in D:
        continue
    sub = [r for r in D[a]["rows"] if r["tgt_style"] != "Minimalism"]
    k = sum(1 for r in sub if r.get("pred_class") == "Minimalism")
    print(f"  {a:20s} {k:3d}/{len(sub):3d} = {k/max(1,len(sub)):.3f}")
PY
