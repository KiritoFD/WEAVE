"""Summarize V10 results at the 750 caliber (2026-11-14 protocol correction).

- V10 arms: runs/v10_local_4070/all/<arm>/eval/eval.json (full 750, run in one dir).
- LD baselines: v9 eval rows merged from dev+test (task-complementary, no
  overlap), cross600 means recomputed from raw per-task rows.

Readout: cross600 (style-transfer subset; same150 is saturated by design).

Run: python -m exp.v10.summarize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V10 = REPO / "runs/v10_local_4070"
V9 = REPO / "runs/v9_local_4070"

METRICS = ["dino_s", "dino_c", "lpips", "target_top1", "target_margin",
           "style_affinity", "clip_s"]


def v10_means(arm: str):
    p = V10 / "all" / arm / "eval/eval.json"
    if not p.exists():
        return None
    summ = json.loads(p.read_text())["summary"]
    m = summ["cross600"]["means"]
    return {k: m[k] for k in METRICS if k in m}, summ["cross600"]["n"]


def v9_merged_means(arm: str):
    """Merge dev+test raw rows for a v9 arm; recompute cross600 means."""
    rows = []
    for split in ("dev", "test"):
        p = V9 / split / arm / "eval/eval.json"
        if p.exists():
            rows.extend(json.loads(p.read_text())["rows"])
    cross = [r for r in rows if r.get("is_cross")]
    if not cross:
        return None
    out = {}
    for k in METRICS:
        vals = [r[k] for r in cross if k in r and isinstance(r[k], (int, float))]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out, len(cross)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="summary_750.json")
    a = ap.parse_args()
    rows = []
    for arm in ("LD0", "LD1", "LD2", "LD3", "LD4"):
        r = v9_merged_means(arm)
        if r:
            rows.append((f"{arm} (v9, merged-750)", r))
    for arm in ("E1", "E2", "E4", "E2_nomask", "E2_mix", "E2_ld4plus"):
        r = v10_means(arm)
        if r:
            rows.append((f"{arm} (v10, 750)", r))
    if not rows:
        print("no eval results found")
        return
    header = f"{'arm':<22}" + "".join(f"{k:>12}" for k in METRICS)
    print(header)
    out = {}
    for name, (s, n) in rows:
        line = f"{name:<22}"
        for k in METRICS:
            v = s.get(k)
            line += f"{v:>12.4f}" if isinstance(v, (int, float)) else f"{'—':>12}"
        print(line)
        out[name] = {"cross_n": n, **s}
    (V10 / a.out).write_text(json.dumps(out, indent=1))
    print("saved ->", V10 / a.out)


if __name__ == "__main__":
    main()
