#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
for a in ("d1","d2"):
    p = Path(f"runs/blog_sbflow/v11/{a}/train_summary.json")
    if not p.exists():
        print(f"  {a}: no summary"); continue
    h = json.loads(p.read_text())["loss_history"]
    dist = [(x["step"], x["dist"]) for x in h if "dist" in x]
    if not dist:
        print(f"  {a}: no dist values"); continue
    print(f"=== {a}: L_dist over training (n={len(dist)}) ===")
    for i in (0, 1, 2, len(dist)//4, len(dist)//2, 3*len(dist)//4, len(dist)-2, len(dist)-1):
        if 0 <= i < len(dist):
            print(f"    step {dist[i][0]:5d}  L_dist={dist[i][1]:.6f}")
    first = sum(d for _, d in dist[:5]) / min(5, len(dist))
    last = sum(d for _, d in dist[-5:]) / min(5, len(dist))
    print(f"  first5 mean={first:.6f}  last5 mean={last:.6f}  "
          f"change={(last-first)/max(first,1e-12)*100:+.1f}%")
PY
