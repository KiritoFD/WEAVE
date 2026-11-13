"""Summarize runs/v8/<arm>/eval/eval.json Cross600 means for given arms.

Usage: python3 -m exp.v8.summarize A3.eval A4.eval C1.l010 ...
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEYS = ["dino_s", "dino_c", "lpips", "target_top1", "dT", "G", "clip_s"]


def main():
    arms = sys.argv[1:]
    print(f"{'arm':<12}" + "".join(f"{k:>10}" for k in KEYS))
    for aid in arms:
        p = REPO / "runs/v8" / aid / "eval" / "eval.json"
        if not p.exists():
            print(f"{aid:<12}  MISSING {p}")
            continue
        d = json.loads(p.read_text())
        rows = [r for r in d["rows"] if r["is_cross"]]
        means = {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}
        print(f"{aid:<12}" + "".join(f"{means[k]:>10.4f}" for k in KEYS)
              + f"  n={len(rows)}")


if __name__ == "__main__":
    main()
