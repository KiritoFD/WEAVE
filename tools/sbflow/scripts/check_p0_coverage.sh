#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("runs/v6_next/amortize/p0_variants.json").read_text())
for v, s in d["variants"].items():
    print(f"  {v}: n_cross={s.get('n_cross')} n_same={s.get('n_same')}")
print()
cache = __import__("torch").load("runs/v5/cache/eval_src_latents.pt", map_location="cpu")
print(f"  eval_src_latents.pt: {len(cache)} stems")
spec = json.loads(Path("eval_protocol/tasks_750.json").read_text())
stems = {t["source_stem"] for t in spec["tasks"]}
print(f"  tasks_750 unique source stems: {len(stems)}")
print(f"  overlap with cache: {len(stems & set(cache))}")
missing = stems - set(cache)
print(f"  missing from cache: {len(missing)}")
for m in list(missing)[:3]:
    print(f"    e.g. {m}")
PY
