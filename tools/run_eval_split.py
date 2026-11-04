"""Evaluation protocol tool: 750 all-pairs primary table + 600/150 explanatory split.

Official protocol (2026-11-04): the ONLY primary metric table is the full
**750 all-pairs** mixture (600 cross-style + 150 same-style, all counted), i.e.
exactly what the paper's main table reports.  600 cross-style and 150
same-style rows are kept for interpretation only and must never be used as a
promotion criterion on their own.

Reference-image selection must be FIXED and recorded inside the result file to
prevent metric drift between runs.

This tool is post-hoc and format-stable:
  Input:  a results JSON produced by utils/run_evaluation.py (or any file with
          a "requests"/"results" list of per-request records containing
          content/style identifiers and metric values), plus
          --reference_policy describing how the style reference was chosen
          (e.g. "first_image_of_target_style_test_dir").
  Output: split report JSON:
          {"all_pairs_750": {...means over all 750...},   # PRIMARY
           "gate_750": {...pass/fail vs 750 thresholds...},
           "main_cross_style": {...means over 600...},     # explanatory
           "identity_same_style": {...means over 150...},  # explanatory
           "reference_selection": {"policy": ..., "note": ...},
           "counts": {...}}

Smoke mode: synthesises 750 fake request records and checks the split math.

Usage:
    python tools/run_eval_split.py --input results.json --reference_policy first_image_of_target_style_test_dir --out results_split.json
    python tools/run_eval_split.py --smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

METRIC_KEYS = ("dino_s", "clip_s", "dino_c", "lpips")

# 750 all-pairs anchor (base_ep10, B/32 + b8 + 8 steps + AdaIN 2.0 + 256px).
ANCHOR_750 = {"dino_s": 0.48587, "dino_c": 0.82902, "clip_s": 0.70940, "lpips": 0.25850}
# Promotion gate expressed on the 750 table (anchor +/- fixed absolute delta).
GATE_750 = {
    "dino_s": {"threshold": 0.5009, "direction": "min"},
    "dino_c": {"threshold": 0.8140, "direction": "min"},
    "clip_s": {"threshold": 0.7194, "direction": "min"},
    "lpips": {"threshold": 0.2685, "direction": "max"},
}


def gate_750(all_pairs: dict) -> dict:
    """Evaluate the 750 promotion gate; all four must pass."""
    passes: dict = {}
    for key, spec in GATE_750.items():
        value = all_pairs.get(key)
        if value is None:
            passes[key] = None
            continue
        if spec["direction"] == "min":
            passes[key] = bool(value >= spec["threshold"])
        else:
            passes[key] = bool(value <= spec["threshold"])
    return {
        "anchor": dict(ANCHOR_750),
        "thresholds": {k: dict(v) for k, v in GATE_750.items()},
        "values": {k: all_pairs.get(k) for k in GATE_750},
        "passes": passes,
        "all_pass": bool(all(v is True for v in passes.values())),
    }


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload.get("requests") or payload.get("results") or payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a 'requests'/'results'/'records' list")
    return records


def is_same_style(rec: dict) -> bool:
    """A request is same-style when content and target style ids match."""
    if "same_style" in rec:
        return bool(rec["same_style"])
    c = rec.get("content_style_id", rec.get("source_style_id"))
    t = rec.get("target_style_id", rec.get("style_id"))
    return c is not None and t is not None and int(c) == int(t)


def _means(records: list[dict]) -> dict:
    out: dict = {}
    for key in METRIC_KEYS:
        vals = [float(r[key]) for r in records if key in r and r[key] is not None]
        if vals:
            out[key] = statistics.fmean(vals)
            out[f"{key}_n"] = len(vals)
    return out


def split_report(records: list[dict], reference_policy: str) -> dict:
    same = [r for r in records if is_same_style(r)]
    cross = [r for r in records if not is_same_style(r)]
    all_pairs = _means(records)
    return {
        "counts": {"total": len(records), "cross_style": len(cross), "same_style": len(same)},
        # PRIMARY table: 750 all-pairs, the paper main-table protocol.
        "all_pairs_750": all_pairs,
        "gate_750": gate_750(all_pairs),
        "primary_table": "all_pairs_750",
        # Explanatory splits only; never a promotion criterion on their own.
        "main_cross_style": _means(cross),
        "identity_same_style": _means(same),
        "reference_selection": {
            "policy": reference_policy,
            "note": "This policy string is part of the metric definition; keep it "
                    "identical across runs. Changing how the style reference image is "
                    "chosen changes DINO-S/CLIP-S and invalidates comparisons.",
        },
        "reporting_rule": "all_pairs_750 is the ONLY primary table and the only "
                          "promotion criterion (600 cross-style + 150 same-style, all "
                          "counted, paper main-table protocol). main_cross_style (600) "
                          "and identity_same_style (150) are explanatory splits and must "
                          "never be reported as the conclusion on their own.",
    }


def run_smoke(out: Path) -> None:
    records = []
    for i in range(750):
        same = i < 150
        sid = i % 5
        cid = sid if same else (sid + 1) % 5
        records.append({
            "content_style_id": cid, "target_style_id": sid,
            "dino_s": 0.4 + (0.08 if same else 0.0), "clip_s": 0.7,
            "dino_c": 0.95 if same else 0.8, "lpips": 0.1 if same else 0.26,
        })
    rep = split_report(records, "first_image_of_target_style_test_dir")
    assert rep["counts"] == {"total": 750, "cross_style": 600, "same_style": 150}, rep["counts"]
    assert rep["primary_table"] == "all_pairs_750"
    # 750 = (150*0.48 + 600*0.40) / 750 = 0.416
    assert abs(rep["all_pairs_750"]["dino_s"] - 0.416) < 1e-9, rep["all_pairs_750"]
    assert abs(rep["main_cross_style"]["dino_s"] - 0.4) < 1e-9
    assert abs(rep["identity_same_style"]["dino_s"] - 0.48) < 1e-9
    # Synthetic record cannot pass the real 750 gate, but the block must exist.
    assert set(rep["gate_750"]["passes"]) == set(GATE_750)
    assert rep["gate_750"]["all_pass"] is False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"smoke": True, "report": rep}, indent=2))
    print(f"[smoke] eval split OK -> {out}; 750 DINO-S={rep['all_pairs_750']['dino_s']:.4f} "
          f"all_pass={rep['gate_750']['all_pass']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--reference_policy", type=str, default="first_image_of_target_style_test_dir")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.smoke:
        run_smoke(out)
        return

    records = load_records(Path(args.input))
    rep = split_report(records, args.reference_policy)
    rep["source_file"] = str(args.input)
    # Carry the evaluation-protocol tags through so the split table states the
    # subset and reference policy it was computed on.
    payload = json.loads(Path(args.input).read_text())
    for key in ("subset", "ref_policy", "protocol"):
        if key in payload:
            rep[key] = payload[key]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    m = rep["all_pairs_750"]
    print(f"750 all-pairs n={rep['counts']['total']} "
          f"dino_s={m.get('dino_s')} dino_c={m.get('dino_c')} "
          f"clip_s={m.get('clip_s')} lpips={m.get('lpips')} "
          f"all_pass={rep['gate_750']['all_pass']} -> {out}")


if __name__ == "__main__":
    main()
