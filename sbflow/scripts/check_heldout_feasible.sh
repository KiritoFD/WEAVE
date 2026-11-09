#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import json
from pathlib import Path

e200 = json.loads(Path("runs/v5/v5_eval200_tasks.json").read_text(encoding="utf-8"))
t750 = json.loads(Path("eval_protocol/tasks_750.json").read_text(encoding="utf-8"))

s200 = {t["src_stem"] for t in e200}
cross = [t for t in t750["tasks"] if t.get("is_cross", t["source_style"] != t["target_style"])]
same  = [t for t in t750["tasks"] if not t.get("is_cross", t["source_style"] != t["target_style"])]
s750 = {Path(t["source_rel_path"]).stem for t in cross}
print(f"eval200 distinct sources          : {len(s200)}")
print(f"tasks_750 cross distinct sources  : {len(s750)}")
print(f"tasks_750 same-style tasks        : {len(same)}")
print(f"eval200 sources inside tasks_750  : {len(s200 & s750)} / {len(s200)}")
disjoint = sorted(s750 - s200)
print(f"SOURCE-DISJOINT sources available : {len(disjoint)}")
tasks_disjoint = [t for t in cross if Path(t["source_rel_path"]).stem in set(disjoint)]
print(f"tasks from those sources          : {len(tasks_disjoint)}")
from collections import Counter
c = Counter(t["target_style"] for t in tasks_disjoint)
print(f"  per target: {dict(c)}")
# if >= 400, a Heldout400 is possible
print()
print(f"VERDICT: source-disjoint Heldout400 is "
      f"{'POSSIBLE' if len(tasks_disjoint) >= 400 else 'NOT possible with 400 tasks'}")
if len(tasks_disjoint) >= 400:
    print(f"  using {len(disjoint)} sources, {len(tasks_disjoint)} tasks available")
PY
