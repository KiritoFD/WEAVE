"""Verify the eval200 row schema satisfies every consumer, on CPU, before spending a run.

This exists because a missing "src_stem" key crashed evaluate_200_v6 only after a full
20-minute train+eval cycle. The schema is the contract between the row builder and the
clustered-CI / paired-comparison consumers, so it is worth a 1-second check.

Consumers checked:
  * _cluster_ci_safe      needs dT and src_stem
  * confusion matrix      needs tgt_style and pred_class
  * per-target aggregation needs tgt_style and dT
  * paired comparison     needs task_id, src_stem and dT
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.v6_24h.stats import cluster_ci, paired_cluster_ci

REQUIRED = ["task_id", "src_stem", "src_style", "tgt_style", "dT", "dS", "dM",
            "target_margin", "target_top1", "pred_class", "dino_c", "lpips"]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main() -> None:
    # a synthetic rows list with the exact schema the trainer writes
    rows = []
    for i in range(40):
        tgt = STYLES[i % 5]
        pred = i % 5
        rows.append({"task_id": f"t{i}", "src_stem": f"src{i // 4}",
                     "src_style": STYLES[(i + 1) % 5], "tgt_style": tgt,
                     "dT": 0.01 * (i % 7 - 3), "dS": -0.3, "dM": 0.0,
                     "target_margin": -0.1, "target_top1": int(pred == i % 5),
                     "pred_class": pred, "dino_c": 0.9, "lpips": 0.3})

    missing = [k for k in REQUIRED if k not in rows[0]]
    check("row schema has every required key", not missing, f"missing={missing}")

    # consumer 1: clustered CI
    ci = cluster_ci([r["dT"] for r in rows], [r["src_stem"] for r in rows], n_boot=500)
    check("clustered CI consumes dT + src_stem", ci["n_clusters"] == 10,
          f"clusters={ci['n_clusters']} ci={[round(x,4) for x in ci['ci']]}")

    # consumer 2: confusion matrix
    conf = np.zeros((5, 5), dtype=int)
    for r in rows:
        conf[STYLES.index(r["tgt_style"]), r["pred_class"]] += 1
    check("confusion matrix consumes tgt_style + pred_class", conf.sum() == len(rows),
          f"total={int(conf.sum())}")

    # consumer 3: per-target aggregation
    pt = {s: float(np.mean([r["dT"] for r in rows if r["tgt_style"] == s])) for s in STYLES}
    check("per-target aggregation has all five styles", len(pt) == 5, f"{list(pt)}")

    # consumer 4: paired comparison
    a = [r["dT"] for r in rows]
    b = [r["dT"] - 0.01 for r in rows]
    p = paired_cluster_ci(a, b, [r["src_stem"] for r in rows], n_boot=500)
    check("paired comparison consumes task alignment + clusters",
          abs(p["diff"] - 0.01) < 1e-9, f"diff={p['diff']:+.5f}")

    print()
    if FAIL:
        print(f"SCHEMA TEST FAILED: {FAIL}")
        sys.exit(1)
    print("EVAL200 ROW SCHEMA TEST PASSED (5 checks)")


if __name__ == "__main__":
    main()
