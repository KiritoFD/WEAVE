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

def load(name):
    p = Path(f"runs/v6_next/eval_750_{name}/eval.json")
    return json.loads(p.read_text())["rows"] if p.exists() else None

def split(rows):
    return ([r for r in rows if r["tgt_style"] != r["src_style"]],
            [r for r in rows if r["tgt_style"] == r["src_style"]])

def summ(rows):
    cross, same = split(rows)
    t1 = np.array([r["target_top1"] for r in cross], float)
    ci = cluster_ci(t1, [r["src_stem"] for r in cross])
    conf = np.zeros((5, 5), dtype=int)
    for r in cross:
        conf[STY.index(r["tgt_style"])][int(r["pred_class"])] += 1
    h = sum(conf[i][MIN] for i in range(5) if i != MIN)
    t = sum(sum(conf[i]) for i in range(5) if i != MIN)
    return {
        "n": len(cross), "top1": float(t1.mean()), "ci": ci,
        "dT": float(np.mean([r["dT"] for r in cross])),
        "dino": float(np.mean([r["dino_c"] for r in cross])),
        "lpips": float(np.mean([r["lpips"] for r in cross])),
        "min_off": h / max(1, t),
        "per_target": {s: float(np.mean([r["target_top1"] for r in cross
                                         if r["tgt_style"] == s])) for s in STY},
        "same_top1": float(np.mean([r["target_top1"] for r in same])),
        "same_dS": float(np.mean([r["dS"] for r in same])),
        "rows": cross,
    }

print("=" * 104)
print("Oracle scored through eval_tiers (same path as every checkpoint)")
print("=" * 104)
S = {}
for n, lab in (("Q2a_embed_style", "Q2a Direct (baseline)"),
               ("oracle_s1", "Q2a + Oracle 1 step")):
    r = load(n)
    if r is None:
        print(f"  {lab}: missing"); continue
    s = summ(r)
    S[n] = s
    print(f"  {lab:26s} n_cross={s['n']}  Top-1={s['top1']*100:5.1f}% "
          f"CI[{s['ci']['ci'][0]*100:.1f},{s['ci']['ci'][1]*100:.1f}]  "
          f"dT={s['dT']:+.4f}  DINO-C={s['dino']:.4f}  LPIPS={s['lpips']:.4f}  "
          f"Min={s['min_off']*100:.1f}%  same={s['same_top1']*100:.1f}%")

if "oracle_s1" in S and "Q2a_embed_style" in S:
    a, b = S["oracle_s1"], S["Q2a_embed_style"]
    cl = [r["src_stem"] for r in b["rows"]]
    ia = [r["task_id"] for r in a["rows"]]
    ib = [r["task_id"] for r in b["rows"]]
    aligned = ia == ib
    print(f"\n  task alignment: {'OK' if aligned else 'MISMATCH'}")
    if aligned:
        t = paired_cluster_ci([r["target_top1"] for r in a["rows"]],
                              [r["target_top1"] for r in b["rows"]], cl)
        d = paired_cluster_ci([r["dino_c"] for r in a["rows"]],
                              [r["dino_c"] for r in b["rows"]], cl)
        print(f"  paired dTop-1 = {t['diff']*100:+.1f}pp "
              f"CI[{t['ci'][0]*100:+.1f},{t['ci'][1]*100:+.1f}] excl0={t['excludes_zero']}")
        print(f"  paired dDINO  = {d['diff']:+.3f} "
              f"CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] excl0={d['excludes_zero']}")
    print("\n  per-target Top-1")
    print(f"  {'model':26s}" + "".join(f"{s[:9]:>11s}" for s in STY))
    for n, lab in (("Q2a_embed_style", "Q2a Direct"), ("oracle_s1", "Oracle 1 step")):
        if n in S:
            print(f"  {lab:26s}" + "".join(f"{S[n]['per_target'][x]*100:10.1f}%"
                                           for x in STY))
PY
