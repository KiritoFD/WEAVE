"""Measure the real-vs-real CSD-SWD / MMD floor per style.

Every SWD number in the reports needs a reference point: how far apart are two samples
of the SAME style? If the real test pool sits 0.024 from the 30-image reference set,
then a generated cloud at 0.026 is near the metric's ceiling and must not be read as
"bad generation".

Uses the frozen evaluator cache only (no models, no GPU):
  eval_protocol/cache/sources_paper_rgb_v1.pt  csd features of the real test pool
  eval_protocol/cache/references_v1_csd_cpu.pt the 30-per-style reference sets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/blog_sbflow/v11/swd_floor.json")
    args = ap.parse_args()

    from tools.sbflow.score_cpu import mmd2, swd

    refs = torch.load(ROOT / "eval_protocol/cache/references_v1_csd_cpu.pt",
                      map_location="cpu", weights_only=False)
    gamma = float(refs["rbf_gamma"])
    src = torch.load(ROOT / "eval_protocol/cache/sources_paper_rgb_v1.pt",
                     map_location="cpu", weights_only=False)

    csd = np.asarray(src["csd"], dtype=np.float32)
    styles = list(src["styles"])
    print(f"[floor] gamma={gamma:.6f} pool={csd.shape} styles={sorted(set(styles))}")

    # Calibration for P3: how often does the CSD prototype classifier label a REAL image
    # of style s as s? That is the ceiling a generated cloud should be compared against,
    # not 1.0.
    proto = None
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from tools.v5.route_c import csd_prototypes
        proto = np.asarray(csd_prototypes(), dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"[floor] prototype calibration unavailable: {e}")

    out = {"mmd_gamma": gamma, "by_style": {}}
    for s in STYLES:
        idx = [i for i, x in enumerate(styles) if x == s]
        if not idx:
            continue
        pool = csd[idx]
        ref = np.asarray(refs["styles"][s]["csd"], dtype=np.float32)
        # real test pool vs reference set: the achievable floor for this metric
        d_swd = swd(pool, ref)
        d_mmd = mmd2(pool, ref, gamma)
        # within-pool split half: an even stricter floor, same sample count as references
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(pool))
        h = len(pool) // 2
        half_swd = swd(pool[perm[:h]], pool[perm[h:2 * h]])
        entry = {"n_pool": int(len(pool)), "n_ref": int(len(ref)),
                 "real_vs_ref_swd": d_swd, "real_vs_ref_mmd": d_mmd,
                 "half_split_swd": half_swd}
        if proto is not None:
            v = pool @ proto.T
            entry["real_style_top1"] = float(np.mean(v.argmax(1) == STYLES.index(s)))
            entry["real_style_margin"] = float(np.mean(
                v[:, STYLES.index(s)] - np.delete(v, STYLES.index(s), axis=1).max(1)))
        out["by_style"][s] = entry
        top1 = entry.get("real_style_top1")
        print(f"  {s:20s} n={len(pool):5d} real-vs-ref SWD={d_swd:.5f} "
              f"MMD={d_mmd:.5f} | half-split SWD={half_swd:.5f}"
              + (f" | real style_top1={top1:.4f}" if top1 is not None else ""))

    vals = [v["real_vs_ref_swd"] for v in out["by_style"].values()]
    out["mean_real_vs_ref_swd"] = float(np.mean(vals)) if vals else None
    out["mean_half_split_swd"] = float(
        np.mean([v["half_split_swd"] for v in out["by_style"].values()])) if vals else None
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[floor] mean real-vs-ref SWD = {out['mean_real_vs_ref_swd']:.5f}")
    print(f"[floor] mean half-split SWD  = {out['mean_half_split_swd']:.5f}")
    print(f"[floor] wrote {p}")


if __name__ == "__main__":
    main()
