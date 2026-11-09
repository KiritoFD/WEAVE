#!/usr/bin/env bash
# Pre-flight for the unattended post-queue run: verify that the task files the smoke test and
# P5 will use actually resolve to existing images. A missing path would only surface after the
# queue has finished and the GPU is free, wasting that window.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import json
from pathlib import Path

for name in ("heldout400_tasks", "same150_tasks", "screen50_tasks"):
    p = Path(f"runs/v6_24h/{name}.json")
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    spec = json.loads(p.read_text())
    tasks = spec["tasks"]
    def sp(t):
        return t.get("source_rel_path") or t.get("src_path")
    paths = {sp(t) for t in tasks}
    missing = [x for x in paths if not Path(x).exists()]
    # also check the per-task fields the evaluator reads
    need = ("task_id", "source_style", "target_style")
    bad = [t for t in tasks[:5] if any(k not in t for k in need)]
    print(f"{name}: {len(tasks)} tasks, {len(paths)} unique sources, "
          f"missing_paths={len(missing)}, bad_schema={len(bad)}")
    if missing:
        print(f"  first missing: {missing[:3]}")
    if bad:
        print(f"  first bad: {sorted(bad[0].keys())}")
    # the first 4 tasks are what the smoke test uses
    first4 = [sp(t) for t in tasks[:4]]
    print(f"  smoke-test first 4 sources exist: "
          f"{[Path(x).exists() for x in first4]}")
PY
