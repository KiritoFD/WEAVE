"""Decompose A1's paradoxical gain across the three content-injection scales.

A1 removes ALL multi-scale content injection and raises dT by +0.0151 while collapsing DINO-C
by 0.1248. S1/S2/S3 remove one scale each, so if the three effects were additive we could say
which scale carries A1's gain.

Prediction from S1 and S2 alone:
    dT_A1 ~= dT_S1 + dT_S2 + dT_S3
    => dT_S3 ~= dT_A1 - dT_S1 - dT_S2

This compares the prediction against the measured S3 and reports the residual. A large residual
means the scales interact, which would be informative in its own right: it would say the
content path is not separable by scale.

Usage
    python3 -m tools.v6_24h.inject_decomposition
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCALES = [("S1_no_inject_64", "64x64"), ("S2_no_inject_32", "32x32"),
          ("S3_no_inject_16", "16x16")]


def ev(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("eval200") or None


def main() -> None:
    a0 = ev(ROOT / "runs/v6/D1_c1_gentle/report.json")
    a1 = ev(ROOT / "runs/v6/A1_no_content/report.json")
    st = json.loads((ROOT / "runs/v6_24h/state.json").read_text())
    singles = {}
    for aid, label in SCALES:
        e = ev(ROOT / f"runs/v6/{aid}/report.json")
        singles[label] = e

    if not a0:
        raise SystemExit("A0 report missing")
    print(f"A0 baseline: dT {a0['mean_dT']:+.4f}  DINO-C {a0['mean_dino_c']:.4f}  "
          f"LPIPS {a0['mean_lpips']:.4f}")
    if a1:
        print(f"A1 (all removed): dT {a1['mean_dT']:+.4f}  "
              f"DINO-C {a1['mean_dino_c']:.4f}  LPIPS {a1['mean_lpips']:.4f}")

    print(f"\n{'single-scale arm':22s}{'dT':>9s}{'delta':>9s}{'DINO-C':>8s}{'delta':>9s}")
    print("-" * 60)
    deltas = {}
    for label in ("64x64", "32x32", "16x16"):
        e = singles[label]
        if not e:
            print(f"{label:22s}{'not run yet':>9s}")
            continue
        dd = e["mean_dT"] - a0["mean_dT"]
        dc = e["mean_dino_c"] - a0["mean_dino_c"]
        deltas[label] = (dd, dc)
        print(f"{label:22s}{e['mean_dT']:+9.4f}{dd:+9.4f}{e['mean_dino_c']:8.4f}{dc:+9.4f}")

    if not a1:
        print("\nA1 report missing; cannot test additivity")
        return
    if len(deltas) < 3:
        print(f"\nonly {len(deltas)}/3 single-scale arms present; waiting on "
              f"{[s for s in ('64x64', '32x32', '16x16') if s not in deltas]}")
        return

    pred = deltas["64x64"][0] + deltas["32x32"][0]
    actual_total = a1["mean_dT"] - a0["mean_dT"]
    implied_s3 = actual_total - pred
    measured_s3 = deltas["16x16"][0]
    resid = measured_s3 - implied_s3

    print(f"\n=== additivity test on dT ===")
    print(f"  A1 total gain                : {actual_total:+.4f}")
    print(f"  S1 (64x64) + S2 (32x32)      : {pred:+.4f}")
    print(f"  implied S3 needed for additivity: {implied_s3:+.4f}")
    print(f"  measured S3                  : {measured_s3:+.4f}")
    print(f"  residual (measured - implied) : {resid:+.4f}")

    print(f"\n=== additivity test on DINO-C ===")
    pred_c = deltas["64x64"][1] + deltas["32x32"][1] + deltas["16x16"][1]
    actual_c = a1["mean_dino_c"] - a0["mean_dino_c"]
    print(f"  A1 total DINO-C change       : {actual_c:+.4f}")
    print(f"  sum of three single-scale    : {pred_c:+.4f}")
    print(f"  residual                     : {actual_c - pred_c:+.4f}")

    print("\nVERDICT:")
    if abs(resid) < 0.005:
        print("  The three scales are approximately ADDITIVE on dT. A1's gain is attributable")
        print("  to the sum of the individual scale effects, and the 16x16 layer carries the")
        print("  predicted share.")
    else:
        print("  The scales are NOT additive: removing them together differs from the sum of")
        print("  the individual removals, so the content path interacts across scales.")
    if abs(actual_c - pred_c) > 0.03:
        print("  DINO-C is strongly NON-additive: the content collapse only appears when")
        print("  multiple scales are removed together.")

    out = ROOT / "runs/v6_24h/inject_decomposition.json"
    out.write_text(json.dumps({
        "a0": {"dT": a0["mean_dT"], "dino": a0["mean_dino_c"], "lpips": a0["mean_lpips"]},
        "a1": {"dT": a1["mean_dT"], "dino": a1["mean_dino_c"], "lpips": a1["mean_lpips"]},
        "single_scale_deltas": {k: {"dT": v[0], "dino": v[1]} for k, v in deltas.items()},
        "implied_s3": implied_s3, "measured_s3": measured_s3, "residual_dT": resid,
        "residual_dino": actual_c - pred_c,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
