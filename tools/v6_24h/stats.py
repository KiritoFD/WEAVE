"""V6 24h plan section 1 - source-clustered paired bootstrap.

The plan requires "按源图聚类的配对 bootstrap" and warns that 40 images per target cannot
support strong claims about small Top-1 changes.

Why the existing `boot_ci` is not enough: it resamples the 200 tasks independently, but
several tasks share the same source image (200 tasks from ~50-150 sources). Tasks that
share a source are correlated, so resampling tasks treats them as independent evidence and
UNDERSTATES the confidence interval.

Two functions:
  cluster_ci(values, clusters)          CI for one arm's mean, resampling CLUSTERS.
  paired_cluster_ci(a, b, clusters)     CI for mean(a) - mean(b) on the SAME clusters,
                                        i.e. the correct interval for an arm-vs-arm claim.

The paired form is what the Go/No-Go rule needs: "dT at least +0.008 vs the matched tier"
is only meaningful with a CI on the DIFFERENCE, not on each arm separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


def _cluster_index(clusters: Sequence[str]) -> list:
    uniq = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(uniq)}
    return [pos[c] for c in clusters], len(uniq)


def cluster_ci(values, clusters, n_boot: int = 4000, seed: int = 0,
               alpha: float = 0.05) -> dict:
    """CI for the mean of `values`, resampling whole clusters with replacement."""
    v = np.asarray(values, dtype=np.float64)
    cid, n_cl = _cluster_index(clusters)
    cid = np.asarray(cid)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_cl, size=n_cl)
        mask = np.isin(cid, pick)
        means[b] = v[mask].mean() if mask.any() else np.nan
    means = means[np.isfinite(means)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "ci": [float(lo), float(hi)],
            "n": int(len(v)), "n_clusters": int(n_cl),
            "boot_mean": float(means.mean()), "boot_std": float(means.std())}


def paired_cluster_ci(a, b, clusters, n_boot: int = 4000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """CI for mean(a) - mean(b), resampling clusters that carry BOTH arms' values.

    `a` and `b` must be aligned per task (same order), and `clusters` labels the source of
    each task. Resampling a cluster therefore resamples the same tasks in both arms, which
    is what makes the interval paired.
    """
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape != bv.shape:
        raise ValueError(f"paired arrays must align: {av.shape} vs {bv.shape}")
    d = av - bv
    cid, n_cl = _cluster_index(clusters)
    cid = np.asarray(cid)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_cl, size=n_cl)
        mask = np.isin(cid, pick)
        means[i] = d[mask].mean() if mask.any() else np.nan
    means = means[np.isfinite(means)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    excl0 = bool(lo > 0 or hi < 0)
    return {"diff": float(d.mean()), "ci": [float(lo), float(hi)],
            "excludes_zero": excl0, "n": int(len(d)), "n_clusters": int(n_cl),
            "boot_std": float(means.std())}


def _rows_of(path: Path, key: str = "eval200") -> list:
    r = json.loads(path.read_text())
    e = r.get(key) or {}
    rows = e.get("rows")
    if not rows:
        raise SystemExit(f"{path} has no per-task rows; it predates row storage. "
                         f"Re-run its eval to enable paired statistics.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="report.json of arm A")
    ap.add_argument("--b", required=True, help="report.json of arm B (reference)")
    ap.add_argument("--metric", default="dT")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    ra = {r["task_id"]: r for r in _rows_of(Path(args.a))}
    rb = {r["task_id"]: r for r in _rows_of(Path(args.b))}
    common = sorted(set(ra) & set(rb))
    if not common:
        raise SystemExit("no common task ids between the two reports")
    a = [ra[k][args.metric] for k in common]
    b = [rb[k][args.metric] for k in common]
    clusters = [ra[k]["src_stem"] for k in common]

    ci_a = cluster_ci(a, clusters, args.n_boot)
    ci_b = cluster_ci(b, clusters, args.n_boot)
    p = paired_cluster_ci(a, b, clusters, args.n_boot)

    print(f"metric = {args.metric}, {len(common)} tasks, "
          f"{p['n_clusters']} source clusters")
    print(f"  A {args.a}:  mean {ci_a['mean']:+.5f}  "
          f"cluster CI [{ci_a['ci'][0]:+.5f}, {ci_a['ci'][1]:+.5f}]")
    print(f"  B {args.b}:  mean {ci_b['mean']:+.5f}  "
          f"cluster CI [{ci_b['ci'][0]:+.5f}, {ci_b['ci'][1]:+.5f}]")
    print(f"  paired diff (A-B): {p['diff']:+.5f}  "
          f"cluster CI [{p['ci'][0]:+.5f}, {p['ci'][1]:+.5f}]  "
          f"excludes_zero={p['excludes_zero']}")


if __name__ == "__main__":
    main()
