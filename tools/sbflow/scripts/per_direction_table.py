"""Recompute the V9 per-direction table with EXPLICIT (source, target) grouping.

The reference `eval_test600.py` aggregates `agg[tag][style]` over rows whose
**target_style == style**. The first V9 report read those keys as if they were
source styles and therefore swapped the two directions. This script recomputes
everything from the raw rows with explicit direction keys, so the table can be
verified independently of the reference aggregator.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
DIRS = [("Impressionism", "Ukiyo_e"), ("Ukiyo_e", "Impressionism")]
KEYS = ("csd_dT", "csd_dS", "target_margin", "target_top1", "lpips_to_input")


def mmd2(gen, real, gamma):
    x = torch.as_tensor(gen, dtype=torch.float32)
    y = torch.as_tensor(real, dtype=torch.float32)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return float("nan")

    def k(a, b):
        return torch.exp(-gamma * torch.cdist(a, b).square())

    xx, yy = k(x, x), k(y, y)
    same = (xx.sum() - xx.diagonal().sum()) / (n * (n - 1))
    realv = (yy.sum() - yy.diagonal().sum()) / (m * (m - 1))
    return float(same + realv - 2 * k(x, y).mean())


def main() -> None:
    refs = torch.load("eval_protocol/cache/references_v1_features.pt", map_location="cpu", weights_only=False)
    gamma = float(refs["rbf_gamma"])

    out = {}
    for f in sorted(glob.glob("runs/blog_sbflow/eval_*/style_scores*.json")):
        if f.endswith("_compact.json"):
            full = f.replace("_compact.json", ".json")
            if os.path.exists(full):
                f = full  # prefer the file that still carries csd_embedding
        name = os.path.basename(os.path.dirname(f)).replace("eval_", "")
        if name in out:
            continue
        d = json.load(open(f))
        entry = {}
        for tag in ("generated", "dcae_noop"):
            rows = [r for r in d["rows"] if r["variant"] == tag]
            per = {}
            for src, tgt in DIRS:
                sub = [r for r in rows if r["source"] == src and r["target"] == tgt]
                if not sub:
                    continue
                res = {k: float(np.mean([r[k] for r in sub])) for k in KEYS}
                res["n"] = len(sub)
                gen = torch.tensor([r["csd_embedding"] for r in sub], dtype=torch.float32) \
                    if "csd_embedding" in sub[0] else None
                res["csd_mmd2"] = mmd2(gen, refs["styles"][tgt]["csd"], gamma) if gen is not None else float("nan")
                per[f"{src}->{tgt}"] = res
            entry[tag] = per
        out[name] = entry

    hdr = (f"{'run':30s} {'variant':10s} {'direction':32s} {'n':>3s} {'top1':>7s} {'margin':>8s} "
           f"{'dT':>8s} {'dS':>8s} {'lpips':>7s} {'mmd2':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for name, e in out.items():
        for tag in ("generated", "dcae_noop"):
            for dkey, r in e[tag].items():
                print(f"{name:30s} {tag:10s} {dkey:32s} {r['n']:3d} {r['target_top1']:7.4f} "
                      f"{r['target_margin']:8.4f} {r['csd_dT']:8.4f} {r['csd_dS']:8.4f} "
                      f"{r['lpips_to_input']:7.4f} {r['csd_mmd2']:7.4f}")
    print()


if __name__ == "__main__":
    main()
