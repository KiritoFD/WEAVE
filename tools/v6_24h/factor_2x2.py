"""Plan section 5 - the 2x2 factor design over (lambda_s, lambda_c).

The plan says A0 and T1 plus L1 and L2 form a 2x2 factorial that explains how much of the
gain comes from stronger style supervision versus weaker content constraint:

              lambda_c=0.240        lambda_c=0.525
  lambda_s=0.35    L1_lowC                A0
  lambda_s=1.20    T1_strong              L2_highS

Reports main effects and the interaction, for dT, DINO-C, LPIPS and per target.

  effect(lambda_s) = mean(rows at 1.20) - mean(rows at 0.35)
  effect(lambda_c) = mean(cols at 0.525) - mean(cols at 0.240)
  interaction      = (L2 - L1) - (T1 - A0)

A non-zero interaction means the two factors are NOT separable, i.e. T1's improvement cannot
be attributed to "more style" alone -- which is exactly the confound the plan warns about.

Usage
    python3 -m tools.v6_24h.factor_2x2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SH = {"Early_Renaissance": "ER", "Impressionism": "Imp", "Minimalism": "Min",
      "Rococo": "Roc", "Ukiyo_e": "Uki"}

# cell -> (lambda_s, lambda_c, candidate report paths)
CELLS = {
    "L1_lowC": (0.35, 0.240, ["runs/v6/L1_lowC/report.json",
                              "runs/v6_24h/L1_lowC/report.json"]),
    "A0": (0.35, 0.525, ["runs/v6/D1_c1_gentle/report.json",
                         "runs/v6/A0_ref/report.json"]),
    "T1_strong": (1.20, 0.240, ["runs/v6/D1_T1_strong/report.json"]),
    "L2_highS": (1.20, 0.525, ["runs/v6/L2_highS/report.json",
                               "runs/v6_24h/L2_highS/report.json"]),
}


def load(paths):
    for p in paths:
        f = ROOT / p
        if f.exists():
            r = json.loads(f.read_text())
            e = r.get("eval200") or {}
            if e:
                return {"path": p, "dT": e.get("mean_dT"),
                        "dino": e.get("mean_dino_c"), "lpips": e.get("mean_lpips"),
                        "sty": e.get("styles_improved"),
                        "per": e.get("per_target_dT") or {}}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="dT", choices=["dT", "dino", "lpips"])
    args = ap.parse_args()

    got = {}
    print(f"{'cell':10s} {'lam_s':>6s} {'lam_c':>6s} {'dT':>9s} {'DINO-C':>8s} "
          f"{'LPIPS':>8s} {'sty+':>5s}  source")
    for name, (ls, lc, paths) in CELLS.items():
        v = load(paths)
        got[name] = v
        if v is None:
            print(f"{name:10s} {ls:6.2f} {lc:6.2f} {'—':>9s} {'—':>8s} {'—':>8s} "
                  f"{'—':>5s}  (not run yet)")
        else:
            print(f"{name:10s} {ls:6.2f} {lc:6.2f} {v['dT']:+9.4f} {v['dino']:8.4f} "
                  f"{v['lpips']:8.4f} {str(v['sty'])+'/5':>5s}  {v['path']}")

    have = {k: v for k, v in got.items() if v}
    if len(have) < 4:
        print(f"\nonly {len(have)}/4 cells available; the 2x2 analysis needs all four.")
        print("waiting on: " + ", ".join(k for k, v in got.items() if not v))
        return

    def cell(name):
        return got[name][args.metric]

    print(f"\n=== 2x2 analysis on {args.metric} ===")
    print(f"{'':22s}{'lam_c=0.240':>14s}{'lam_c=0.525':>14s}{'row effect':>14s}")
    for ls, rn, ln in ((0.35, "L1_lowC", "A0"), (1.20, "T1_strong", "L2_highS")):
        row_eff = cell(ln) - cell(rn)
        print(f"{'lam_s='+format(ls,'.2f'):22s}{cell(rn):+14.5f}{cell(ln):+14.5f}"
              f"{row_eff:+14.5f}")
    col_eff_lo = cell("T1_strong") - cell("L1_lowC")
    col_eff_hi = cell("L2_highS") - cell("A0")
    print(f"{'column effect':22s}{col_eff_lo:+14.5f}{col_eff_hi:+14.5f}")

    eff_s = 0.5 * ((cell("T1_strong") - cell("L1_lowC"))
                   + (cell("L2_highS") - cell("A0")))
    eff_c = 0.5 * ((cell("A0") - cell("L1_lowC")) + (cell("L2_highS") - cell("T1_strong")))
    # CORRECT 2x2 interaction: effect of one factor at the two levels of the other.
    #   (y11 - y10) - (y01 - y00)  with y00=L1, y10=T1, y01=A0, y11=L2
    # An earlier version computed (y11 - y00) - (y10 - y01), which is a difference of the two
    # DIAGONALS, not the interaction, and gave +0.0105 instead of the correct -0.0029.
    inter = (cell("L2_highS") - cell("T1_strong")) - (cell("A0") - cell("L1_lowC"))
    inter_check = (cell("L2_highS") - cell("A0")) - (cell("T1_strong") - cell("L1_lowC"))

    print(f"\n  main effect of lambda_s (0.35 -> 1.20): {eff_s:+.5f}")
    print(f"  main effect of lambda_c (0.240 -> 0.525): {eff_c:+.5f}")
    print(f"  INTERACTION (y11-y10)-(y01-y00): {inter:+.5f}")
    print(f"    equivalent form (y11-y01)-(y10-y00): {inter_check:+.5f}  "
          f"(must match)")
    if abs(inter - inter_check) > 1e-9:
        print("    WARNING: the two interaction forms disagree, cells are inconsistent")
    print()
    if abs(inter) > max(abs(eff_s), abs(eff_c)) * 0.5:
        print("  -> interaction is large relative to the main effects: the two factors are")
        print("     NOT separable, so T1's gain cannot be attributed to 'more style' alone.")
    else:
        print("  -> interaction is small relative to the main effects: the two factors are")
        print("     approximately separable, so the main effects can be read independently.")

    print(f"\n{'per-target ' + args.metric:22s}" + "".join(f"{SH[s]:>9s}" for s in STYLES))
    for name in ("L1_lowC", "A0", "T1_strong", "L2_highS"):
        per = got[name]["per"]
        print(f"{name:22s}" + "".join(f"{per.get(s, float('nan')):+9.4f}" for s in STYLES))


if __name__ == "__main__":
    main()
