#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("eval_protocol/tasks_750.json").read_text())
print("top keys:", list(d.keys()))
t = d.get("tasks") or []
cross = [x for x in t if x.get("is_cross")]
same = [x for x in t if not x.get("is_cross")]
print(f"total={len(t)} cross={len(cross)} same={len(same)}")
print(f"declared: total={d.get('num_tasks_total')} cross={d.get('num_tasks_cross')} "
      f"same={d.get('num_tasks_same')} sources={d.get('num_sources')}")
print("\nper-source-style:")
from collections import Counter
c = Counter(x["source_style"] for x in cross)
for k, v in c.items():
    print(f"  {k:22s} {v}")
print("\nper-target-style (cross):")
c2 = Counter(x["target_style"] for x in cross)
for k, v in c2.items():
    print(f"  {k:22s} {v}")
print("\nunique cross sources:", len({x['source_stem'] for x in cross}))
p0 = t[0]["source_rel_path"]
print("\nsample rel path:", p0)
print("exists from repo root:", Path(p0).exists())
print("exists from asset root:", (Path("/mnt/i/Latent_Style") / p0).exists())
PY
