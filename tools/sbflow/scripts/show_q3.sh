#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MODELS = ["Q2a_embed_style", "P2_G1b_bf16", "P2_G0_bf16", "L11_k1024_dim16"]

def load(tier, m):
    for base in ("runs/v6_next", "runs/v6_24h"):
        p = Path(f"{base}/eval_{tier}_{m}/eval.json")
        if p.exists():
            return json.loads(p.read_text())
    return None

print("=" * 104)
print("Q3 INDEPENDENT EVALUATION")
print("=" * 104)
for tier in ("heldout400", "same150"):
    print(f"\n### {tier}")
    print(f"  {'model':22s}{'n':>5s}{'dT':>9s}{'dS':>9s}{'Top-1':>8s}{'margin':>9s}"
          f"{'DINO-C':>8s}{'LPIPS':>8s}{'sty':>5s}")
    for m in MODELS:
        d = load(tier, m)
        if d is None:
            print(f"  {m:22s}  not evaluated"); continue
        print(f"  {m:22s}{d['n']:5d}{d['mean_dT']:+9.4f}{d['mean_dS']:+9.4f}"
              f"{d['mean_top1']:8.3f}{d['mean_margin']:+9.4f}"
              f"{d['mean_dino_c']:8.4f}{d['mean_lpips']:8.4f}"
              f"{d['styles_improved']}/5")
        ci = d.get("dT_cluster_ci") or {}
        if ci:
            print(f"  {'':22s}   dT CI [{ci['ci'][0]:+.4f}, {ci['ci'][1]:+.4f}]"
                  f" ({ci['n_clusters']} clusters)")

print("\n" + "=" * 104)
print("PER-TARGET Top-1 (Heldout400)")
print("=" * 104)
print(f"  {'model':22s}" + "".join(f"{s[:9]:>11s}" for s in STY))
for m in MODELS:
    d = load("heldout400", m)
    if d is None:
        continue
    rows = d["rows"]
    cells = []
    for s in STY:
        sub = [r for r in rows if r["tgt_style"] == s]
        cells.append(sum(r["target_top1"] for r in sub) / max(1, len(sub)))
    print(f"  {m:22s}" + "".join(f"{c:11.3f}" for c in cells))

print("\n" + "=" * 104)
print("PER-TARGET Top-1 (Same150)")
print("=" * 104)
print(f"  {'model':22s}" + "".join(f"{s[:9]:>11s}" for s in STY))
for m in MODELS:
    d = load("same150", m)
    if d is None:
        continue
    rows = d["rows"]
    cells = []
    for s in STY:
        sub = [r for r in rows if r["tgt_style"] == s]
        cells.append(sum(r["target_top1"] for r in sub) / max(1, len(sub)))
    print(f"  {m:22s}" + "".join(f"{c:11.3f}" for c in cells))

print("\n" + "=" * 104)
print("NON-TARGET MINIMALISM MISCLASSIFICATION (Heldout400)")
print("=" * 104)
for m in MODELS:
    d = load("heldout400", m)
    if d is None:
        continue
    sub = [r for r in d["rows"] if r["tgt_style"] != "Minimalism"]
    k = sum(1 for r in sub if r.get("pred_class") == "Minimalism")
    print(f"  {m:22s} {k:3d}/{len(sub):3d} = {k/max(1,len(sub)):.3f}")
PY
