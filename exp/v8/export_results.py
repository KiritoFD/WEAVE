"""Export per-arm eval means (Cross600 + Same150) to JSON for reporting.

Usage: python3 -m exp.v8.export_results > runs/v8/results_summary.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEYS = ["dino_s", "dino_c", "lpips", "target_top1", "target_margin", "dT", "G",
        "clip_s", "style_affinity"]


def means(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}


def main():
    out = {}
    for d in sorted((REPO / "runs/v8").iterdir()):
        p = d / "eval" / "eval.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        rows = data.get("rows", [])
        cross = [r for r in rows if r.get("is_cross")]
        same = [r for r in rows if not r.get("is_cross")]
        out[d.name] = {
            "n_total": len(rows), "n_cross": len(cross), "n_same": len(same),
            "cross": means(cross) if cross else None,
            "same": means(same) if same else None,
        }
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
