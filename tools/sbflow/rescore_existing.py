"""Recompute the V10 Quick200 aggregates with the corrected per-target SWD/MMD.

V10's `overall` block compared all 200 mixed-target images against ONE style's
reference set, so the reported overall CSD-SWD/MMD was not a five-target
distribution-matching metric. This rewrites the aggregate from the stored per-row
CSD embeddings without re-running any model, and reports old vs new.

Usage (on 3060):
    python3 -m sbflow.rescore_existing --root runs/blog_sbflow/suite40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.sbflow.score_cpu import aggregate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/blog_sbflow/suite40")
    ap.add_argument("--pattern", default="score_*_quick200")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    root = repo / args.root if not Path(args.root).is_absolute() else Path(args.root)
    refs = torch.load(repo / "eval_protocol/cache/references_v1_csd_cpu.pt",
                      map_location="cpu", weights_only=False)
    gamma = float(refs["rbf_gamma"])
    print(f"[rescore] gamma={gamma:.6f} root={root}")

    rows_out = []
    for f in sorted(root.glob(f"{args.pattern}/style_scores.json")):
        d = json.loads(f.read_text())
        rows = d["rows"]
        old = d["aggregate"]["overall"]
        new = aggregate(rows, refs, gamma)
        bt = new["by_target"]
        rows_out.append({
            "arm": f.parent.name,
            "old_swd": old.get("csd_swd"), "new_swd": new["overall"]["csd_swd"],
            "old_mmd": old.get("csd_mmd2_unbiased"), "new_mmd": new["overall"]["csd_mmd2_unbiased"],
            "top1": new["overall"]["target_top1"], "dT": new["overall"]["csd_dT"],
            "per_target_swd": {k: v["csd_swd"] for k, v in bt.items()},
            "n_targets": new["overall"].get("n_targets"),
        })
        if args.write:
            d["aggregate"] = new
            d["aggregate_v11_note"] = ("overall SWD/MMD is now the equal-weight mean over "
                                       "the five target styles (V11 fix)")
            (f.parent / "style_scores_v11.json").write_text(
                json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'arm':22s} {'old_swd':>9s} {'new_swd':>9s} {'delta':>9s} "
          f"{'old_mmd':>9s} {'new_mmd':>9s} {'top1':>7s} {'dT':>8s}")
    for r in rows_out:
        print(f"{r['arm']:22s} {r['old_swd']:9.5f} {r['new_swd']:9.5f} "
              f"{r['new_swd'] - r['old_swd']:+9.5f} {r['old_mmd']:9.5f} "
              f"{r['new_mmd']:9.5f} {r['top1']:7.4f} {r['dT']:+8.4f}")

    if len(rows_out) >= 2:
        base = rows_out[0]
        print("\nrelative to the first arm:")
        for r in rows_out[1:]:
            dswd = (r["new_swd"] - base["new_swd"]) / max(abs(base["new_swd"]), 1e-12)
            print(f"  {r['arm']:22s} new_swd {dswd*100:+.2f}%  dT {r['dT'] - base['dT']:+.4f}")

    out = root / "rescore_v11.json"
    out.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[rescore] wrote {out}")


if __name__ == "__main__":
    main()
