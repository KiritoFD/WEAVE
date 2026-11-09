"""Verify how eval_test600 aggregates per-style numbers: by target_style, not source."""
import json
from collections import defaultdict

import numpy as np

P = "runs/blog_sbflow/eval_ImpUki_OT0.5_step4000/style_scores_compact.json"
d = json.load(open(P))
rows = [r for r in d["rows"] if r["variant"] == "generated"]

print("total generated rows:", len(rows))
for key in ("Impressionism", "Ukiyo_e"):
    sub = [r for r in rows if r["target"] == key]
    srcs = sorted({r["source"] for r in sub})
    print(f"\nagg key '{key}' -> rows with target=='{key}': n={len(sub)}, sources={srcs}")
    print("  top1   :", round(float(np.mean([r["target_top1"] for r in sub])), 4))
    print("  margin :", round(float(np.mean([r["target_margin"] for r in sub])), 4))
    print("  csd_dT :", round(float(np.mean([r["csd_dT"] for r in sub])), 4))
    print("  agg says:", {k: round(v, 4) for k, v in d["aggregate"]["generated"][key].items()
                          if k in ("target_top1", "target_margin", "csd_dT", "csd_mmd2_unbiased")})

# Cross-check by explicit direction
print("\n--- explicit direction breakdown ---")
for src, tgt in [("Impressionism", "Ukiyo_e"), ("Ukiyo_e", "Impressionism")]:
    sub = [r for r in rows if r["source"] == src and r["target"] == tgt]
    print(f"{src} -> {tgt}: n={len(sub)} top1={np.mean([r['target_top1'] for r in sub]):.4f} "
          f"margin={np.mean([r['target_margin'] for r in sub]):.4f} dT={np.mean([r['csd_dT'] for r in sub]):.4f}")
