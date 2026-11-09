"""V6 24h plan section 1 - build the Screen50 task subset.

Screen50 = 50 cross-style tasks, 10 per target style, with a fixed source/task-id seed.
It exists for coarse screening of inference knobs (the multiplication stage) and must NOT
be used to claim final gains; Val200 remains the comparison set.

Deterministic: same seed -> same 50 tasks, so a knob sweep is comparable across runs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v6.train_eval_v6 import STYLES, V5_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-target", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="runs/v6_24h/screen50_tasks.json")
    args = ap.parse_args()

    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    by_tgt = {s: [] for s in STYLES}
    for t in tasks:
        by_tgt[t["tgt_style"]].append(t)

    picked = []
    for s in STYLES:
        pool = sorted(by_tgt[s], key=lambda x: x["task_id"])
        rng.shuffle(pool)
        picked.extend(sorted(pool[: args.per_target], key=lambda x: x["task_id"]))

    # distinct source images actually used (for cache warm-up planning)
    srcs = {t["src_stem"] for t in picked}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": {"name": "Screen50", "seed": args.seed,
                                            "per_target": args.per_target,
                                            "n_tasks": len(picked),
                                            "note": "coarse screening only; Val200 is "
                                                    "the comparison set"},
                               "tasks": picked}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[screen50] {len(picked)} tasks, {len(srcs)} distinct sources -> {out}")
    for s in STYLES:
        n = sum(1 for t in picked if t["tgt_style"] == s)
        print(f"  {s:20s} {n}")
    if len(picked) != args.per_target * len(STYLES):
        raise SystemExit("not enough tasks per target in the source pool")


if __name__ == "__main__":
    main()
