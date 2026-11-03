"""Evaluation protocol tool: 600 cross-style main set + 150 same-style identity report.

Review doc requirement (section 四/五): the 750-request eval mixes two tasks.
Cross-style transfer is the MAIN task; same-style requests only measure
Identity Preservation and must be reported separately. Reference-image
selection must be FIXED and recorded inside the result file to prevent metric
drift between runs.

This tool is post-hoc and format-stable:
  Input:  a results JSON produced by utils/run_evaluation.py (or any file with
          a "requests"/"results" list of per-request records containing
          content/style identifiers and metric values), plus
          --reference_policy describing how the style reference was chosen
          (e.g. "first_image_of_target_style_test_dir").
  Output: split report JSON:
          {"main_cross_style": {...means over 600...},
           "identity_same_style": {...means over 150...},
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
    return {
        "counts": {"total": len(records), "cross_style": len(cross), "same_style": len(same)},
        "main_cross_style": _means(cross),
        "identity_same_style": _means(same),
        "reference_selection": {
            "policy": reference_policy,
            "note": "This policy string is part of the metric definition; keep it "
                    "identical across runs. Changing how the style reference image is "
                    "chosen changes DINO-S/CLIP-S and invalidates comparisons.",
        },
        "reporting_rule": "main_cross_style is the primary table (target: 600 requests); "
                          "identity_same_style (target: 150) is a separate Identity "
                          "Preservation report, never averaged into the main table.",
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
    assert abs(rep["main_cross_style"]["dino_s"] - 0.4) < 1e-9
    assert abs(rep["identity_same_style"]["dino_s"] - 0.48) < 1e-9
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"smoke": True, "report": rep}, indent=2))
    print(f"[smoke] eval split OK -> {out}; main DINO-S={rep['main_cross_style']['dino_s']}")


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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"cross-style n={rep['counts']['cross_style']} same-style n={rep['counts']['same_style']} -> {out}")


if __name__ == "__main__":
    main()
