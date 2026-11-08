"""Build the V9 comparison table from compact score JSONs."""
from __future__ import annotations

import glob
import json
import os

ROWS = []


def main() -> None:
    for f in sorted(glob.glob("runs/blog_sbflow/eval_*/style_scores_compact.json")):
        d = json.load(open(f))
        name = os.path.basename(os.path.dirname(f)).replace("eval_", "")
        g = d["aggregate"]["generated"]
        n = d["aggregate"]["dcae_noop"]
        ROWS.append(
            {
                "run": name,
                "overall": g["cross_overall"],
                "imp": g.get("Impressionism", {}),
                "uki": g.get("Ukiyo_e", {}),
                "noop": n["cross_overall"],
            }
        )

    hdr = f"{'run':32s} {'top1':>7s} {'margin':>8s} {'dT':>8s} {'lpips':>7s} {'T_Imp':>7s} {'T_Uki':>7s} {'MMD_Imp':>8s} {'MMD_Uki':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in ROWS:
        o, i, u = r["overall"], r["imp"], r["uki"]
        print(
            f"{r['run']:32s} {o['target_top1']:7.4f} {o['target_margin']:8.4f} "
            f"{o['csd_dT']:8.4f} {o['lpips_to_input']:7.4f} "
            f"{i.get('target_top1', float('nan')):7.4f} {u.get('target_top1', float('nan')):7.4f} "
            f"{i.get('csd_mmd2_unbiased', float('nan')):8.4f} {u.get('csd_mmd2_unbiased', float('nan')):8.4f}"
        )
    noop = ROWS[0]["noop"]
    print("\nDCAE no-op baseline:")
    print(
        f"{'dcae_noop':32s} {noop['target_top1']:7.4f} {noop['target_margin']:8.4f} "
        f"{noop['csd_dT']:8.4f} {noop['lpips_to_input']:7.4f}"
    )


if __name__ == "__main__":
    main()
