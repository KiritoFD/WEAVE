#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path

def check(tag, p):
    p = Path(p)
    if not p.exists():
        print(f"  {tag}: missing"); return
    d = json.loads(p.read_text())
    rows = d.get("rows") or []
    if not rows:
        print(f"  {tag}: no rows"); return
    k0 = list(rows[0].keys())
    has = "pred_class" in k0
    non_min = [r for r in rows if r["tgt_style"] != "Minimalism"]
    if has:
        k = sum(1 for r in non_min if r["pred_class"] == "Minimalism")
    else:
        # fall back: reconstruct argmax from the confusion matrix is lossy; use target_top1 only
        k = -1
    print(f"  {tag}: n={len(rows)} pred_class_in_rows={has} "
          f"nonMin={len(non_min)} min_hits={k}")
    if not has:
        print(f"      row keys: {k0}")

check("Q2a Heldout400", "runs/v6_next/eval_heldout400_Q2a_embed_style/eval.json")
check("Q2a Same150", "runs/v6_next/eval_same150_Q2a_embed_style/eval.json")
check("L11 Heldout400", "runs/v6_24h/eval_heldout400_L11/eval.json")

# and the P0 A0 rows, which store argmax explicitly
d = json.loads(Path("runs/v6_next/amortize/p0_variants.json").read_text())
r0 = d["rows"]["A0"]
cross = [r for r in r0 if r["is_cross"]]
nm = [r for r in cross if r["tgt_style"] != "Minimalism"]
k = sum(1 for r in nm if r["argmax"] == "Minimalism")
print(f"\n  P0 A0 (Test600): cross={len(cross)} nonMin={len(nm)} "
      f"min_hits={k} rate={k/max(1,len(nm)):.3f}")
from collections import Counter
print("  A0 predicted-class distribution over non-Min targets:")
for kk, v in Counter(r["argmax"] for r in nm).most_common():
    print(f"      {kk:22s} {v}")
print("\n  A0 per-target argmax breakdown:")
for st in ("Early_Renaissance", "Impressionism", "Rococo", "Ukiyo_e"):
    sub = [r for r in cross if r["tgt_style"] == st]
    print(f"    target={st:20s} n={len(sub)} argmax counts="
          f"{dict(Counter(r['argmax'] for r in sub))}")
PY
