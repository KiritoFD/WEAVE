#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import json
from pathlib import Path
tasks = json.loads(Path("runs/v5/v5_eval200_tasks.json").read_text(encoding="utf-8"))
stems = sorted({t["src_stem"] for t in tasks})
print(f"eval200 tasks={len(tasks)}  distinct sources={len(stems)}")
print(f"  -> same-style reconstructions actually evaluated = {len(stems)}")
print(f"  (the code calls it 'same50'; the plan asks for Same150)")
print(f"  source styles present: {sorted({t['src_style'] for t in tasks})}")
# how many tasks per source
from collections import Counter
c = Counter(t["src_stem"] for t in tasks)
print(f"  tasks per source: min={min(c.values())} max={max(c.values())} "
      f"mean={sum(c.values())/len(c):.2f}")
# tasks_750 for Heldout400 planning
t7 = json.loads(Path("eval_protocol/tasks_750.json").read_text())
cross = [t for t in t7["tasks"] if t.get("is_cross", t["source_style"] != t["target_style"])]
same = [t for t in t7["tasks"] if not t.get("is_cross", t["source_style"] != t["target_style"])]
print(f"\ntasks_750: total={len(t7['tasks'])} cross={len(cross)} same={len(same)}")
print(f"  cross distinct sources={len({t.get('source_rel_path') for t in cross})}")
PY
