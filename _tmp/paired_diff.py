"""S2 - S1 paired diff on the 600 cross-style rows; source-stratified bootstrap."""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

RR = Path("/mnt/i/Latent_Style/pseudo_gt_24h/eval_render")
CELLS = {"S1": RR / "S1_A_P5_final", "S2": RR / "S2_E7_final"}
METRICS = ["dino_s", "dino_c", "clip_s", "lpips", "csd_dM", "swd"]
N_BOOT = 10000
SEED = 20261106


def load(cell: Path):
    rows = json.loads((cell / "v4_eval750_rows.json").read_text())
    out = {}
    for r in rows:
        if not r.get("cross"):
            continue
        dM = float(r["csd_dT"]) - float(r["csd_dS"])
        rec = {k: float(r[k]) for k in ("dino_s", "dino_c", "clip_s", "lpips", "swd")}
        rec["csd_dM"] = dM
        out[(r["src"], r["tgt_style"])] = (r["src"], rec)
    return out


s1, s2 = load(CELLS["S1"]), load(CELLS["S2"])
keys = sorted(set(s1) & set(s2))
print(f"paired_rows={len(keys)} only_S1={len(set(s1) - set(s2))} only_S2={len(set(s2) - set(s1))}")
by_src: dict[str, list[tuple[float, ...]]] = {}
for k in keys:
    src = s1[k][0]
    by_src.setdefault(src, []).append(tuple(s2[k][1][m] - s1[k][1][m] for m in METRICS))
print(f"sources={len(by_src)}")

rng = random.Random(SEED)
srcs = list(by_src)
boot = {m: [] for m in METRICS}
boot_med = {m: [] for m in METRICS}
all_d = {m: [v[i] for vals in by_src.values() for v in vals] for i, m in enumerate(METRICS)}
for _ in range(N_BOOT):
    pick = [rng.choice(srcs) for _ in srcs]
    sample = [v for s in pick for v in by_src[s]]
    for i, m in enumerate(METRICS):
        col = [v[i] for v in sample]
        boot[m].append(sum(col) / len(col))
        boot_med[m].append(statistics.median(col))


def ci(vals):
    vals = sorted(vals)
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return lo, hi


print(f"{'metric':<8} {'mean':>10} {'95%CI':>22} {'median':>10} {'med95CI':>22} {'frac>0':>8}")
for m in METRICS:
    d = all_d[m]
    lo, hi = ci(boot[m])
    mlo, mhi = ci(boot_med[m])
    print(f"{m:<8} {sum(d) / len(d):>+10.6f} [{lo:+.6f}, {hi:+.6f}] "
          f"{statistics.median(d):>+10.6f} [{mlo:+.6f}, {mhi:+.6f}] {sum(v > 0 for v in d) / len(d):>8.4f}")

payload = {
    "cells": {"A": str(CELLS["S1"]), "B": str(CELLS["S2"])},
    "n_paired": len(keys), "n_sources": len(by_src), "n_boot": N_BOOT, "seed": SEED,
    "metrics": {
        m: {
            "mean": sum(all_d[m]) / len(all_d[m]),
            "ci95": list(ci(boot[m])),
            "median": statistics.median(all_d[m]),
            "median_ci95": list(ci(boot_med[m])),
            "frac_gt0": sum(v > 0 for v in all_d[m]) / len(all_d[m]),
        } for m in METRICS
    },
}
Path("/mnt/i/Latent_Style/WEAVE/runs/v4/paired_diff_S2_minus_S1.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2))
print("wrote runs/v4/paired_diff_S2_minus_S1.json")
