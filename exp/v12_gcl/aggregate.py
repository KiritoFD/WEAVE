"""V12-GCL aggregate: source-clustered paired bootstrap vs the LD4 baseline.

Reads runs/v12_gcl/all/<arm>/eval/eval.json rows (750 in one dir) and the LD4
merged rows (v9 dev+test eval.json). Bootstrap resamples the 150 source stems
(painting-level clusters), 4000 replicates, reports mean diffs with CI95 and
the pre-registered success criteria (plan §5):
  expanded frontier := (|dDINO-C| <= .003 and dAffinity >= +.003)
                    or (|dAffinity| <= .002 and dDINO-C >= +.008)
Run: python -m exp.v12_gcl.aggregate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
V12 = REPO / "runs/v12_gcl"
V9 = REPO / "runs/v9_local_4070"
METRICS = ["dino_s", "dino_c", "lpips", "target_top1", "target_margin",
           "style_affinity"]


def merged_rows(arm: str, root: Path, splits=("dev", "test")):
    rows = []
    for sp in splits:
        p = root / sp / arm / "eval/eval.json"
        if p.exists():
            rows.extend(json.loads(p.read_text())["rows"])
    return rows


def rows_by_source(rows):
    out = {}
    for r in rows:
        out.setdefault(r["src_stem"], []).append(r)
    return out


def arm_rows_all(arm: str):
    p = V12 / "all" / arm / "eval/eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["rows"]


def mean_metric(rows, key):
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


def paired_bootstrap(rows_a, rows_b, key, n_boot=4000, seed=0):
    """rows_a/b keyed by task_id; returns (mean_a, mean_b, mean_diff, ci95_diff)."""
    da = {r["task_id"]: r for r in rows_a}
    db = {r["task_id"]: r for r in rows_b}
    common = sorted(set(da) & set(db))
    if not common:
        return None
    srcs = {}
    for tid in common:
        srcs.setdefault(da[tid]["src_stem"], []).append(tid)
    src_list = sorted(srcs)
    va = torch.tensor([da[t][key] for t in common], dtype=torch.float64)
    vb = torch.tensor([db[t][key] for t in common], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [src_list[torch.randint(0, len(src_list), (1,), generator=g).item()]
                for _ in src_list]
        ia = [t for s in pick for t in srcs[s] if t in da]
        ib = [t for s in pick for t in srcs[s] if t in db]
        ma = sum(da[t][key] for t in ia) / len(ia)
        mb = sum(db[t][key] for t in ib) / len(ib)
        diffs.append(mb - ma)
    lo, hi = torch.quantile(torch.tensor(diffs, dtype=torch.float64),
                            0.025).item(), \
        torch.quantile(torch.tensor(diffs, dtype=torch.float64), 0.975).item()
    ma = sum(da[t][key] for t in common) / len(common)
    mb = sum(db[t][key] for t in common) / len(common)
    return {"mean_a": round(ma, 4), "mean_b": round(mb, 4),
            "diff": round(mb - ma, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="LD4")
    ap.add_argument("--out", default="summary_gcl.json")
    a = ap.parse_args()

    base_rows = merged_rows(a.baseline, V9)
    assert base_rows, "LD4 baseline rows not found"
    arms = ["G0S", "G1", "G2O", "G2", "G2S"]
    base_cross = [r for r in base_rows if r["is_cross"]]
    summary = {}
    for arm in arms:
        rows = arm_rows_all(arm)
        if rows is None:
            continue
        cross = [r for r in rows if r["is_cross"]]
        stats = {"cross_n": len(cross)}
        for k in METRICS:
            stats[k] = round(mean_metric(cross, k), 4)
        # paired bootstrap vs the LD4 baseline on shared tasks
        da = {r["task_id"]: r for r in base_cross}
        shared = [r for r in cross if r["task_id"] in da]
        pb = {}
        for k in METRICS:
            r = paired_bootstrap(da_rows := base_cross, rows, k) if False else \
                paired_bootstrap_taskkey(da, {t["task_id"]: t for t in shared}, k)
            if r:
                pb[k] = r
        stats["paired_vs_LD4"] = pb
        # pre-registered expansion criteria (plan §5)
        pb_c = pb.get("dino_c", {"diff": 9, "ci95": [9, 9]})
        pb_a = pb.get("style_affinity", {"diff": 9, "ci95": [9, 9]})
        d_c, ci_c = pb_c["diff"], pb_c["ci95"]
        d_a, ci_a = pb_a["diff"], pb_a["ci95"]
        stats["expanded_frontier"] = {
            "dDINO-C_le_.003_and_dAff_ge_.003":
                bool(abs(d_c) <= 0.003 and d_a >= 0.003 and ci_a[0] > 0),
            "dAff_le_.002_and_dDINOC_ge_.008":
                bool(abs(d_a) <= 0.002 and d_c >= 0.008 and ci_c[0] > 0),
        }
        summary[arm] = stats
    (V12 / a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("saved ->", V12 / a.out)


def paired_bootstrap_taskkey(da, db, key, n_boot=4000, seed=0):
    common = sorted(set(da) & set(db))
    if not common:
        return None
    srcs = {}
    for tid in common:
        srcs.setdefault(da[tid]["src_stem"], []).append(tid)
    src_list = sorted(srcs)
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [src_list[torch.randint(0, len(src_list), (1,), generator=g).item()]
                for _ in src_list]
        ia = [t for s in pick for t in srcs[s] if t in da]
        ib = [t for s in pick for t in srcs[s] if t in db]
        ma = sum(da[t][key] for t in ia) / len(ia)
        mb = sum(db[t][key] for t in ib) / len(ib)
        diffs.append(mb - ma)
    lo = torch.quantile(torch.tensor(diffs, dtype=torch.float64), 0.025).item()
    hi = torch.quantile(torch.tensor(diffs, dtype=torch.float64), 0.975).item()
    ma = sum(da[t][key] for t in common) / len(common)
    mb = sum(db[t][key] for t in common) / len(common)
    return {"mean_a": round(ma, 4), "mean_b": round(mb, 4),
            "diff": round(mb - ma, 4), "ci95": [round(lo, 4), round(hi, 4)]}


if __name__ == "__main__":
    main()
