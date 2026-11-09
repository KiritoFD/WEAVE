"""Is the confusion matrix merely uninformative, or systematically ANTI-target?

"Top-1 below chance" can mean two very different things:
  (a) uninformative -- predictions are uniform random, so the diagonal equals 200/5 = 40;
  (b) anti-target   -- the generated image is more likely to be classified as some OTHER
                       style than as its own target, so the diagonal is far BELOW 40.

These have different implications. (a) means the output carries no style signal; (b) means it
carries a signal that points the wrong way, which is a much stronger statement and would
suggest the transformation actively misleads the classifier.

This computes, per arm: the diagonal total against the uniform expectation, a chi-square
statistic against the uniform matrix, and the column bias (which prototype attracts mass).

Usage
    python3 -m tools.v6_24h.confusion_analysis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SH = {"Early_Renaissance": "ER", "Impressionism": "Imp", "Minimalism": "Min",
      "Rococo": "Roc", "Ukiyo_e": "Uki"}


def main() -> None:
    state = json.loads((ROOT / "runs/v6_24h/state.json").read_text())
    arms = []
    for name, v in state.items():
        p = ROOT / f"runs/v6/{name}/report.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text()).get("eval200") or {}
        conf = e.get("confusion_target_vs_pred")
        if not conf:
            continue
        arms.append((name, np.asarray(conf, dtype=float)))

    if not arms:
        print("no arms with a confusion matrix yet")
        return

    n_cls = len(STYLES)
    print(f"{'arm':16s}{'n':>5s}{'diag':>7s}{'uniform':>9s}{'diag/unif':>11s}"
          f"{'chi2':>10s}  column bias (max/min)")
    print("-" * 92)
    rows = []
    for name, C in arms:
        n = int(C.sum())
        diag = float(np.trace(C))
        unif = n / n_cls
        # chi-square against the uniform matrix: rows fixed by target, cols uniform
        exp = np.full_like(C, n / (n_cls * n_cls))
        chi2 = float(((C - exp) ** 2 / exp).sum())
        col = C.sum(0)
        bias = f"{int(col.max())}/{int(col.min())} (max {SH[STYLES[int(col.argmax())]]}, " \
               f"min {SH[STYLES[int(col.argmin())]]})"
        ratio = diag / unif
        rows.append({"arm": name, "n": n, "diag": diag, "uniform": unif,
                     "diag_over_uniform": ratio, "chi2": chi2,
                     "col_totals": col.tolist()})
        print(f"{name:16s}{n:5d}{diag:7.0f}{unif:9.1f}{ratio:11.2f}{chi2:10.1f}  {bias}")

    print(f"\nuniform-random expectation: diagonal = n/{n_cls} = {rows[0]['n'] / n_cls:.0f} "
          f"per arm, ratio 1.00")
    ratios = [r["diag_over_uniform"] for r in rows]
    print(f"observed diag/uniform ratio: min {min(ratios):.2f}, max {max(ratios):.2f}, "
          f"mean {np.mean(ratios):.2f}")

    print("\nVERDICT:")
    if max(ratios) < 0.5:
        print("  Every arm is far BELOW the uniform-random diagonal, i.e. the generated image")
        print("  is more likely to be classified as a DIFFERENT style than as its own target.")
        print("  This is stronger than 'uninformative': the transformation is systematically")
        print("  anti-target in CSD space, not merely style-free.")
    elif max(ratios) < 0.9:
        print("  Arms sit below the uniform-random diagonal: predictions are not independent")
        print("  of the target in a neutral way, they lean away from it.")
    else:
        print("  At least one arm reaches the uniform-random diagonal, so the output is")
        print("  uninformative rather than anti-target.")

    # column bias across arms: is there a prototype that always attracts mass?
    print(f"\ncolumn totals per arm (which prototype attracts predictions):")
    print(f"{'arm':16s}" + "".join(f"{SH[s]:>7s}" for s in STYLES))
    for name, C in arms:
        print(f"{name:16s}" + "".join(f"{int(x):7d}" for x in C.sum(0)))
    mean_col = np.mean([C.sum(0) for _, C in arms], axis=0)
    print(f"{'mean':16s}" + "".join(f"{x:7.1f}" for x in mean_col))
    print(f"  uniform would be {rows[0]['n'] / n_cls:.1f} per column")

    # ---- decisive check: is the prediction just the SOURCE style? ----
    # On a cross-style task the source style is by construction NOT the target, so a
    # near-identity operator would show a near-zero diagonal AND near-uniform columns --
    # exactly the pattern above. That would make "anti-target" a misreading of "still source".
    print(f"\n{'arm':16s}{'pred==target':>14s}{'pred==source':>14s}{'pred==other':>13s}")
    print("-" * 60)
    src_hits, tgt_hits = [], []
    for name, v in state.items():
        p = ROOT / f"runs/v6/{name}/report.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text()).get("eval200") or {}
        rws = e.get("rows")
        if not rws:
            continue
        n = len(rws)
        t = sum(1 for r in rws if r["pred_class"] == STYLES.index(r["tgt_style"]))
        s = sum(1 for r in rws if r["pred_class"] == STYLES.index(r["src_style"]))
        o = n - t - s
        src_hits.append(s / n)
        tgt_hits.append(t / n)
        print(f"{name:16s}{t / n:14.3f}{s / n:14.3f}{o / n:13.3f}")

    if src_hits:
        print(f"\nmean pred==source: {np.mean(src_hits):.3f}   "
              f"mean pred==target: {np.mean(tgt_hits):.3f}")
        print("\nREVISED VERDICT:")
        if np.mean(src_hits) > 0.5:
            print("  The prediction is usually the SOURCE style. On a cross-style task that")
            print("  style is by construction not the target, so a near-zero diagonal with")
            print("  near-uniform columns is exactly what a NEAR-IDENTITY operator produces.")
            print("  'Anti-target' is therefore a MISREADING of 'still the source style'.")
            print("  The correct statement: in CSD space the output is still recognised as")
            print("  the source style, i.e. the transformation does not move the image far")
            print("  enough to change its style identity.")
        elif np.mean(src_hits) > 0.3:
            print("  Predictions lean toward the source style but not decisively; the output")
            print("  is partway between source identity and something else.")
        else:
            print("  Predictions are neither source nor target: the output has lost the")
            print("  source identity without acquiring the target one, i.e. it is genuinely")
            print("  off-manifold rather than near-identity.")

    out = ROOT / "runs/v6_24h/confusion_analysis.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
