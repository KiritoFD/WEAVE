"""Q0: verify the P2 arms and produce the paired statistics the acceptance criteria need.

Nothing here trains. It answers four questions the review requires before any further GPU time:

  1. Are all three P2 arms complete and internally consistent (checkpoint, report, n_params)?
  2. Relative to G0, is the Top-1 gain real under a paired source-clustered bootstrap?
  3. Does the target margin move in the same direction as Top-1?
  4. Did the Minimalism attractor grow? Reported as an ABSOLUTE off-diagonal rate with a CI, not
     only as a ratio to G0, because a ratio is unstable when G0's count is small.
  5. Do the arms actually differ, i.e. did the ranking term change anything at all? This is the
     guard against the generator simply ignoring the new loss.

Per-target results for Early_Renaissance, Rococo and Ukiyo_e are printed separately, and the
non-target Minimalism misclassification rate is reported for every arm.

Usage
    python3 -m tools.v6_next.q0_verify
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.v6.train_eval_v6 import STYLES
from tools.v6_24h.stats import paired_cluster_ci, cluster_ci

ARMS = ["P2_G0_bf16", "P2_G1a_bf16", "P2_G1b_bf16"]
MIN_IDX = STYLES.index("Minimalism")
HARD = ["Early_Renaissance", "Rococo", "Ukiyo_e"]


def load(name: str) -> dict | None:
    p = ROOT / f"runs/v6/{name}/report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def rows(name: str) -> list[dict] | None:
    d = load(name)
    if not d:
        return None
    e = d.get("eval200") or {}
    return e.get("rows")


def offdiag_min(rs: list[dict]) -> tuple[float, int, int]:
    """Fraction of tasks whose PREDICTED style is Minimalism while the TARGET is not."""
    non_min = [r for r in rs if r["tgt_style"] != "Minimalism"]
    hits = [r for r in non_min if r.get("pred_class") == "Minimalism"]
    return (len(hits) / max(1, len(non_min)), len(hits), len(non_min))


def main() -> None:
    print("=" * 100)
    print("Q0  P2 verification")
    print("=" * 100)

    # ---- 1. completeness ----
    print("\n[1] arm completeness")
    ok = True
    for a in ARMS:
        d = load(a)
        if d is None:
            print(f"  {a:18s} MISSING report")
            ok = False
            continue
        e = d.get("eval200") or {}
        ck = ROOT / f"runs/v6/{a}/{a}.pt"
        print(f"  {a:18s} params={d.get('n_params'):>9,} "
              f"ckpt={'yes' if ck.exists() else 'NO'} "
              f"steps={d.get('steps')} seed={d.get('seed')} "
              f"dT={e.get('mean_dT'):+.4f} Top1={_top1(e):.3f} "
              f"DINO-C={e.get('mean_dino_c'):.4f} LPIPS={e.get('mean_lpips'):.4f}")
    if not ok:
        print("\n  ABORT: not all arms present; nothing to compare")
        return

    # ---- 2. headline table ----
    print("\n[2] Val200 headline (model selection uses Val200 only)")
    print(f"  {'arm':18s}{'dT':>9s}{'Top-1':>8s}{'margin':>9s}{'DINO-C':>8s}"
          f"{'LPIPS':>8s}{'sty+':>6s}{'same dS':>9s}")
    base = load("P2_G0_bf16")["eval200"]
    for a in ARMS:
        e = load(a)["eval200"]
        print(f"  {a:18s}{e['mean_dT']:+9.4f}{_top1(e):8.3f}{e['mean_dM']:+9.4f}"
              f"{e['mean_dino_c']:8.4f}{e['mean_lpips']:8.4f}"
              f"{str(e['styles_improved'])+'/5':>6s}{e.get('same50_mean_dS', 0):+9.4f}")

    # ---- 3. paired comparison vs G0 ----
    print("\n[3] paired source-clustered comparison vs G0_ctrl")
    r0 = rows("P2_G0_bf16")
    t0 = _per_task(r0, "dT")
    for a in ARMS[1:]:
        ra = rows(a)
        ta = _per_task(ra, "dT")
        ci = paired_cluster_ci(ta, t0, _clusters(r0))          # G1 minus G0
        t1_0 = np.array([int(r["target_top1"]) for r in r0], float)
        t1_a = np.array([int(r["target_top1"]) for r in ra], float)
        ci_t1 = paired_cluster_ci(t1_a, t1_0, _clusters(r0))   # G1 minus G0
        print(f"  {a:18s} d(dT)={ci['diff']:+.4f} CI [{ci['ci'][0]:+.4f}, {ci['ci'][1]:+.4f}] "
              f"excl0={ci['excludes_zero']}")
        print(f"  {'':18s} d(Top-1)={ci_t1['diff']:+.4f} "
              f"CI [{ci_t1['ci'][0]:+.4f}, {ci_t1['ci'][1]:+.4f}] "
              f"excl0={ci_t1['excludes_zero']}")

    # ---- 4. per-target, with the hard three separate ----
    print("\n[4] per-target Top-1 (hard targets listed separately)")
    hdr = f"  {'arm':18s}" + "".join(f"{s[:9]:>11s}" for s in STYLES)
    print(hdr)
    for a in ARMS:
        ra = rows(a)
        cells = []
        for s in STYLES:
            sub = [r for r in ra if r["tgt_style"] == s]
            cells.append(sum(r["target_top1"] for r in sub) / max(1, len(sub)))
        print(f"  {a:18s}" + "".join(f"{c:11.3f}" for c in cells))
    print("\n  hard targets explicitly:")
    for a in ARMS:
        ra = rows(a)
        txt = []
        for s in HARD:
            sub = [r for r in ra if r["tgt_style"] == s]
            txt.append(f"{s[:9]}={sum(r['target_top1'] for r in sub)/max(1,len(sub)):.3f}")
        print(f"  {a:18s} " + "  ".join(txt))

    # ---- 5. Minimalism attractor, absolute rate with CI ----
    print("\n[5] non-target Minimalism misclassification (absolute rate, clustered CI)")
    print("     rate = P(predicted == Minimalism | target != Minimalism)")
    for a in ARMS:
        ra = rows(a)
        rate, k, n = offdiag_min(ra)
        sub = [r for r in ra if r["tgt_style"] != "Minimalism"]
        flags = [1.0 if r.get("pred_class") == "Minimalism" else 0.0 for r in sub]
        cl = [r["src_stem"] for r in sub]
        ci = cluster_ci(flags, cl) if flags else None
        print(f"  {a:18s} {k:3d}/{n:3d} = {rate:.3f}"
              + (f"  CI [{ci['ci'][0]:.3f}, {ci['ci'][1]:.3f}]"
                 f"  ({ci['n_clusters']} clusters)" if ci else ""))

    # ---- 6. did the arms actually differ? ----
    print("\n[6] arm-to-arm divergence (guard against the generator ignoring the new loss)")
    for i in range(len(ARMS)):
        for j in range(i + 1, len(ARMS)):
            a, b = ARMS[i], ARMS[j]
            pa = _per_task(rows(a), "dT")
            pb = _per_task(rows(b), "dT")
            same = float(np.mean(np.isclose(pa, pb, atol=1e-9)))
            print(f"  {a:18s} vs {b:18s} identical_fraction={same:.3f} "
                  f"max|diff|={float(np.max(np.abs(pa-pb))):.5f}")
    print("\n  (identical_fraction near 1.0 would mean the ranking term had no effect at all)")

    # ---- 7. verdict ----
    print("\n[7] verdict against the six acceptance criteria")
    e0 = load("P2_G0_bf16")["eval200"]
    for a in ARMS[1:]:
        e = load(a)["eval200"]
        ra, r0 = rows(a), rows("P2_G0_bf16")
        d_top1 = _top1(e) - _top1(e0)
        ci = paired_cluster_ci(np.array([int(r["target_top1"]) for r in ra], float),
                               np.array([int(r["target_top1"]) for r in r0], float),
                               _clusters(r0))                 # G1 minus G0
        crit1 = d_top1 >= 0.05 and ci["excludes_zero"]
        crit2 = (e["mean_dM"] - e0["mean_dM"]) * d_top1 > 0
        crit3 = e["mean_dino_c"] >= 0.85
        rate_a, _, _ = offdiag_min(ra)
        rate_0, _, _ = offdiag_min(r0)
        crit5 = rate_a <= max(0.05, rate_0 * 1.2)
        print(f"  {a}")
        print(f"    1 Top-1 +5pp & CI excl 0 : {crit1}  (d={d_top1:+.3f}, "
              f"CI [{ci['ci'][0]:+.3f},{ci['ci'][1]:+.3f}])")
        print(f"    2 margin same direction  : {crit2}")
        print(f"    3 DINO-C >= 0.85         : {crit3}  ({e['mean_dino_c']:.4f})")
        print(f"    5 Minimalism not inflated: {crit5}  ({rate_a:.3f} vs G0 {rate_0:.3f})")
        print(f"    => {'PASS' if (crit1 and crit2 and crit3 and crit5) else 'FAIL'}")


def _top1(e: dict) -> float:
    pt = e.get("per_target_top1") or {}
    return float(np.mean(list(pt.values()))) if pt else 0.0


def _per_task(rs: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rs], dtype=float)


def _clusters(rs: list[dict]) -> list[str]:
    return [r["src_stem"] for r in rs]


if __name__ == "__main__":
    main()
