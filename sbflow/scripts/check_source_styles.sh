#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

for name, path in (("Heldout400", "runs/v6_24h/heldout400_tasks.json"),
                   ("Test600", "eval_protocol/tasks_750.json"),
                   ("Same150", "runs/v6_24h/same150_tasks.json")):
    d = json.loads(Path(path).read_text())
    tasks = d["tasks"]
    cross = [t for t in tasks if t.get("is_cross", True)]
    key = "source_style"
    src = Counter(t[key] for t in cross)
    srcs = {t["source_stem"]: t[key] for t in tasks}
    print(f"\n{name}: cross={len(cross)} unique_sources={len(srcs)}")
    print(f"  source styles (per task): {dict(src)}")
    print(f"  source styles (per unique source): "
          f"{dict(Counter(srcs.values()))}")
PY
