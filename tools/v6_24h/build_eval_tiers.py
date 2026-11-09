"""V6 24h plan section 1 - build the Heldout400 and Same150 evaluation tiers.

Heldout400: 400 cross-style tasks whose SOURCE images are disjoint from the 50 sources used
by Val200. Verified feasible: tasks_750 has 150 cross sources, Val200 uses 50 of them, so
exactly 100 sources / 400 tasks remain, balanced 80 per target style. This makes it a real
source-disjoint held-out set for the final 2-3 models, unlike a random re-split.

Caveat to keep in the report: V6.1 already looked at Heldout400 historically, so it is not
a set that never influenced research decisions. It IS source-disjoint from Val200, which is
what makes it a valid check against Val200 overfitting.

Same150: the 150 same-style tasks from tasks_750. The plan forbids substituting code-level
identity for these; they must be run through the network so self-consistency is measured.

Usage
    python3 -m tools.v6_24h.build_eval_tiers
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/v6_24h"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tasks", type=int, default=400)
    args = ap.parse_args()

    e200 = json.loads((ROOT / "runs/v5/v5_eval200_tasks.json").read_text(encoding="utf-8"))
    t750 = json.loads((ROOT / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))
    val_sources = {t["src_stem"] for t in e200}

    cross, same = [], []
    for t in t750["tasks"]:
        is_cross = t.get("is_cross", t["source_style"] != t["target_style"])
        (cross if is_cross else same).append(t)

    def stem_of(t):
        return Path(t["source_rel_path"]).stem

    disjoint = sorted({stem_of(t) for t in cross} - val_sources)
    held = [t for t in cross if stem_of(t) in set(disjoint)]
    # deterministic, balanced: sort by (target, task_id) then take round-robin per target
    by_t = {}
    for t in sorted(held, key=lambda x: (x["target_style"], x["task_id"])):
        by_t.setdefault(t["target_style"], []).append(t)
    picked, i = [], 0
    while len(picked) < min(args.max_tasks, len(held)):
        added = False
        for s in sorted(by_t):
            if i < len(by_t[s]) and len(picked) < args.max_tasks:
                picked.append(by_t[s][i])
                added = True
        if not added:
            break
        i += 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "heldout400_tasks.json").write_text(json.dumps({
        "protocol": {"name": "Heldout400", "n_tasks": len(picked),
                     "n_sources": len({stem_of(t) for t in picked}),
                     "source_disjoint_from_val200": True,
                     "val200_sources": len(val_sources),
                     "caveat": "V6.1 saw this set historically; it is source-disjoint from "
                               "Val200 but not pristine."},
        "tasks": picked}, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "same150_tasks.json").write_text(json.dumps({
        "protocol": {"name": "Same150", "n_tasks": len(same),
                     "note": "must be run through the NETWORK; code-level identity is "
                             "forbidden as a substitute for measured self-consistency"},
        "tasks": same}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[tiers] Heldout400: {len(picked)} tasks from "
          f"{len({stem_of(t) for t in picked})} source-disjoint sources")
    print(f"  per target: {dict(Counter(t['target_style'] for t in picked))}")
    print(f"[tiers] Same150: {len(same)} tasks")
    print(f"  per style: {dict(Counter(t['source_style'] for t in same))}")
    print(f"  -> {OUT}/heldout400_tasks.json, same150_tasks.json")


if __name__ == "__main__":
    main()
