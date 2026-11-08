"""Combine ANY number of two-domain pair evaluations into one honest summary.

`merge_test600.py` requires exactly 10 unordered pairs (full 600). This lighter
tool works with 1..10 pairs and always reports the *coverage* explicitly, so a
single-pair result can never be mistaken for the full frozen Test600.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
KEYS = ("csd_dT", "csd_dS", "target_margin", "target_top1", "lpips_to_input")


def _mmd2(gen, real, gamma):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    refs = torch.load(repo / "eval_protocol/cache/references_v1_features.pt",
                      map_location="cpu", weights_only=False)
    gamma = float(refs["rbf_gamma"])

    rows, pairs, models = [], set(), []
    for f in args.results:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        a, b = d["protocol"]["source_style"], d["protocol"]["target_style"]
        pairs.add(frozenset((a, b)))
        models.append(str(f))
        for r in d["rows"]:
            rows.append({**r, "pair": f"{a}<->{b}", "source_file": str(f)})

    ids_by_variant = {}
    for tag in ("generated", "dcae_noop"):
        ids = [r["task_id"] for r in rows if r["variant"] == tag]
        ids_by_variant[tag] = {"n": len(ids), "unique": len(set(ids))}

    out = {
        "protocol": {
            "frozen": "WEAVE eval_protocol/tasks_750.json (cross-style subset)",
            "n_unordered_pairs": len(pairs),
            "n_pairs_required_for_full_600": 10,
            "coverage_of_600_cross": len(set(t for t in ids_by_variant["generated"].values())) and ids_by_variant["generated"]["unique"] / 600.0,
            "mmd_gamma": gamma,
            "models": models,
            "task_counts": ids_by_variant,
        },
        "variants": {},
    }

    for tag in ("generated", "dcae_noop"):
        subset = [r for r in rows if r["variant"] == tag]
        entry = {"overall": {}, "per_target": {}}
        entry["overall"] = {k: float(np.mean([r[k] for r in subset])) for k in KEYS}
        for style in STYLES:
            s_rows = [r for r in subset if r["target"] == style]
            if not s_rows:
                continue
            res = {k: float(np.mean([r[k] for r in s_rows])) for k in KEYS}
            gen = torch.tensor([r["csd_embedding"] for r in s_rows], dtype=torch.float32)
            res["csd_mmd2_unbiased"] = _mmd2(gen, refs["styles"][style]["csd"], gamma)
            res["n"] = len(s_rows)
            entry["per_target"][style] = res
        out["variants"][tag] = entry

    # Delta of generated minus no-op, same task set.
    gen_rows = {r["task_id"]: r for r in rows if r["variant"] == "generated"}
    noop_rows = {r["task_id"]: r for r in rows if r["variant"] == "dcae_noop"}
    common = sorted(set(gen_rows) & set(noop_rows))
    delta = {}
    if common:
        for k in KEYS:
            g = np.array([gen_rows[t][k] for t in common])
            n = np.array([noop_rows[t][k] for t in common])
            delta[k] = {
                "mean_generated": float(g.mean()),
                "mean_dcae_noop": float(n.mean()),
                "mean_delta": float((g - n).mean()),
                "win_rate": float((g > n).mean()) if k != "lpips_to_input" else float((g < n).mean()),
            }
    out["generated_minus_dcae_noop"] = {"n_common": len(common), "metrics": delta}

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
