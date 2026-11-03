"""Merge metrics.csv (CLIP/LPIPS) + dino_metrics.csv into one results.json.

The evaluation pipeline writes two CSV files with different metric sets; the
split report tool (tools/run_eval_split.py) expects a single record list.  This
step is format-stable: rows are joined on (src_style, tgt_style, src_image,
gen_image) and the output keeps the canonical field names used downstream
(content_style_id, target_style_id, same_style, src_image, gen_image, dino_s,
dino_c, clip_s, lpips).

Usage:
    python tools/merge_eval_metrics.py --metrics DIR/metrics.csv \
        --dino DIR/dino_metrics.csv --out DIR/results.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

KEY = ("src_style", "tgt_style", "src_image", "gen_image")


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _key(row: dict) -> tuple:
    return tuple(str(row.get(k, "")).strip() for k in KEY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics.csv from run_evaluation.py")
    ap.add_argument("--dino", required=True, help="dino_metrics.csv from compute_dino_metrics.py")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--protocol", default="",
        help="Optional eval_protocol.json written by run_evaluation.py; copied into results.json "
             "so subset / ref_policy are recorded alongside the numbers.",
    )
    args = ap.parse_args()

    base = _read(Path(args.metrics))
    dino = {_key(r): r for r in _read(Path(args.dino))}

    records, missing = [], 0
    for row in base:
        k = _key(row)
        d = dino.get(k)
        if d is None:
            missing += 1
            continue
        src_style, tgt_style = row["src_style"].strip(), row["tgt_style"].strip()
        records.append({
            "content_style_id": src_style,
            "target_style_id": tgt_style,
            "same_style": src_style == tgt_style,
            "src_image": row["src_image"].strip(),
            "gen_image": row["gen_image"].strip(),
            "dino_s": float(d["dino_s"]),
            "dino_c": float(d["dino_c"]),
            "dino_structure": float(d["dino_structure"]),
            "clip_s": float(row["clip_style"]),
            "clip_dir": float(row.get("clip_dir", 0.0) or 0.0),
            "lpips": float(row["content_lpips"]),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_metrics": str(args.metrics),
        "source_dino": str(args.dino),
        "results": records,
    }
    if str(args.protocol).strip():
        proto_path = Path(args.protocol)
        payload["protocol"] = json.loads(proto_path.read_text())
        payload["subset"] = payload["protocol"].get("subset")
        payload["ref_policy"] = payload["protocol"].get("ref_policy")
    out.write_text(json.dumps(payload, indent=2))
    print(f"merged {len(records)} records (missing dino rows: {missing}) -> {out}")


if __name__ == "__main__":
    main()
