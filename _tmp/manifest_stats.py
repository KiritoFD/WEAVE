"""Summarize paired-cache manifest by (method, patch_spec) to detect mixed sessions."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/pseudo_gt_24h/pairs")

for arm in sys.argv[1:] or ["base", "positive", "reverse"]:
    p = ROOT / arm / "manifest.json"
    if not p.exists():
        print(f"[{arm}] no manifest")
        continue
    recs = json.loads(p.read_text(encoding="utf-8"))["records"]
    c = Counter((r.get("method"), r.get("patch_spec")) for r in recs)
    print(f"[{arm}] n={len(recs)} combos={len(c)}")
    for k, v in c.items():
        print(f"  method={k[0]} spec={k[1]} -> {v}")
    vers = Counter((r["source_id"], r["donor_id"]) for r in recs)
    print(f"[{arm}] pair_versions: " + " ".join(f"v{k}={sum(1 for x in vers.values() if x == k)}"
                                               for k in sorted(set(vers.values()))))
    print(f"[{arm}] z0_original={sum(1 for r in recs if r.get('z0_is_original'))} "
          f"z1_original={sum(1 for r in recs if r.get('z1_is_original'))}")
