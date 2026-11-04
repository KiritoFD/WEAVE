"""Print 750 / 600 / 150 means for one eval_fast output directory.

Usage:
    python3 tools/12h/summarize_results.py runs/12h/anchors/base_ep10_750_512
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

KEYS = ("dino_s", "dino_c", "clip_s", "lpips")


def means(rows: list[dict]) -> dict[str, float]:
    return {k: round(st.mean(float(r[k]) for r in rows), 5) for k in KEYS}


def main() -> int:
    run_dir = Path(sys.argv[1])
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows = payload["results"]
    proto = payload.get("protocol", {})
    print(f"# {run_dir}")
    print(f"# image_size={proto.get('image_size')} eval_batch={proto.get('eval_batch')} "
          f"steps={proto.get('steps')} clip={proto.get('clip_backbone')} dino={proto.get('dino_backbone')} "
          f"backbone={proto.get('backbone')}")
    print(f"n_all={len(rows)}")
    cross = [r for r in rows if not r.get("same_style")]
    same = [r for r in rows if r.get("same_style")]
    for name, group in (("ALL", rows), ("CROSS", cross), ("SAME", same)):
        if group:
            print(f"{name:6s} n={len(group):4d} " + " ".join(f"{k}={v:.5f}" for k, v in means(group).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
