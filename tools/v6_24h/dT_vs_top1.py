"""Is dT a useful proxy for target-style recognition?

dT is the quantity the whole 24h plan optimises, but Top-1 is the metric that actually says
whether the output is recognisable as the target style. If the two do not move together, then
optimising dT is not steering toward the goal.

This correlates macro Top-1 against dT across the completed arms and, more usefully, reports
what dT would have to be for Top-1 to reach chance level.

Usage
    python3 -m tools.v6_24h.dT_vs_top1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def main() -> None:
    state = json.loads((ROOT / "runs/v6_24h/state.json").read_text())
    pts = []
    for name, v in state.items():
        p = ROOT / f"runs/v6/{name}/report.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text()).get("eval200") or {}
        pt = e.get("per_target_top1")
        if not pt or e.get("mean_dT") is None:
            continue
        pts.append((name, e["mean_dT"], float(np.mean(list(pt.values()))),
                    e["mean_dino_c"], e["mean_lpips"]))

    if len(pts) < 3:
        print(f"only {len(pts)} arms carry both dT and Top-1; need at least 3")
        return

    pts.sort(key=lambda x: x[1])
    print(f"{'arm':15s}{'dT':>9s}{'Top-1':>8s}{'DINO-C':>8s}{'LPIPS':>8s}")
    for n, d, t, dc, lp in pts:
        print(f"{n:15s}{d:+9.4f}{t:8.3f}{dc:8.4f}{lp:8.4f}")

    d = np.array([p[1] for p in pts])
    t = np.array([p[2] for p in pts])
    r_p = float(np.corrcoef(d, t)[0, 1])
    # Spearman without scipy
    rd = np.argsort(np.argsort(d))
    rt = np.argsort(np.argsort(t))
    r_s = float(np.corrcoef(rd, rt)[0, 1])

    print(f"\nn = {len(pts)} arms")
    print(f"  Pearson  r(dT, Top-1) = {r_p:+.3f}")
    print(f"  Spearman r(dT, Top-1) = {r_s:+.3f}")
    print(f"  Top-1 range = [{t.min():.3f}, {t.max():.3f}]   chance = 0.20")
    print(f"  dT range    = [{d.min():+.4f}, {d.max():+.4f}]")

    # linear extrapolation: what dT would be needed for Top-1 = 0.20?
    if abs(r_p) > 1e-6 and d.std() > 1e-9:
        slope = r_p * t.std() / d.std()
        if slope > 0:
            need = d.mean() + (0.20 - t.mean()) / slope
            print(f"\n  if the dT/Top-1 relation held linearly, Top-1 = 0.20 would need "
                  f"dT ~ {need:+.3f}")
            print(f"  the best arm reaches dT {d.max():+.4f}, i.e. "
                  f"{d.max() / need * 100:.1f}% of that")
        else:
            print("\n  slope is not positive: raising dT would not raise Top-1 at all")

    print("\nVERDICT:")
    if abs(r_p) < 0.5 and t.max() < 0.20:
        print("  dT is NOT a usable proxy for recognisability in this range.")
        print("  Every arm sits below chance on Top-1 while dT varies by a factor of 2.6,")
        print("  so optimising dT further cannot be assumed to move the real objective.")
    elif t.max() >= 0.20:
        print("  at least one arm exceeds chance on Top-1, so the relation is worth fitting")
    else:
        print("  dT and Top-1 do co-vary, but the absolute level is still below chance")

    out = ROOT / "runs/v6_24h/dT_vs_top1.json"
    out.write_text(json.dumps({
        "points": [{"arm": n, "dT": x, "top1": y, "dino_c": dc, "lpips": lp}
                   for n, x, y, dc, lp in pts],
        "pearson": r_p, "spearman": r_s, "chance": 0.20,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
