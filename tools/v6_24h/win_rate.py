"""Per-task win rate and its source-clustered CI.

The mean dT can be dragged around by a few large outliers, so the fraction of tasks whose dT
is positive is a more robust statement about whether an arm helps consistently. With only ~50
source clusters, that fraction also needs a clustered interval rather than a binomial one.

This matters for interpretation: if the win rate is well above 0.5 with a tight clustered CI,
then the dT improvement is a REAL and consistent directional shift -- the problem is that the
shift is too small to make the output recognisable, not that the measurement is noise.

Usage
    python3 -m tools.v6_24h.win_rate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.v6_24h.stats import _cluster_index


def win_rate_ci(flags, clusters, n_boot: int = 4000, seed: int = 0,
                alpha: float = 0.05) -> dict:
    v = np.asarray(flags, dtype=np.float64)
    cid, n_cl = _cluster_index(clusters)
    cid = np.asarray(cid)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_cl, size=n_cl)
        m = np.isin(cid, pick)
        out[i] = v[m].mean() if m.any() else np.nan
    out = out[np.isfinite(out)]
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"rate": float(v.mean()), "ci": [float(lo), float(hi)],
            "n": int(len(v)), "n_clusters": int(n_cl),
            "above_half": bool(lo > 0.5)}


def main() -> None:
    state = json.loads((ROOT / "runs/v6_24h/state.json").read_text())
    rows_out = []
    print(f"{'arm':15s}{'win rate':>10s}{'cluster CI':>22s}{'n':>5s}{'clusters':>9s}"
          f"{'  >0.5?'}")
    for name, v in state.items():
        p = ROOT / f"runs/v6/{name}/report.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text()).get("eval200") or {}
        rows = e.get("rows")
        if not rows:
            print(f"{name:15s}{'no rows (predates row storage)':>45s}")
            continue
        flags = [1 if r["dT"] > 0 else 0 for r in rows]
        cl = [r["src_stem"] for r in rows]
        r = win_rate_ci(flags, cl)
        rows_out.append({"arm": name, **r})
        ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        print(f"{name:15s}{r['rate']:10.3f}{ci:>22s}{r['n']:5d}{r['n_clusters']:9d}"
              f"  {r['above_half']}")

    if rows_out:
        best = max(rows_out, key=lambda x: x["rate"])
        print(f"\nhighest win rate: {best['arm']} {best['rate']:.3f} "
              f"CI [{best['ci'][0]:.3f}, {best['ci'][1]:.3f}]")
        print("\nInterpretation: a win rate clearly above 0.5 with a clustered CI that")
        print("excludes 0.5 means the directional shift is real and consistent across tasks.")
        print("Combined with Top-1 near 0.03, that means the shift is RELIABLE BUT TOO SMALL")
        print("-- not that dT is measurement noise.")
        out = ROOT / "runs/v6_24h/win_rate.json"
        out.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
