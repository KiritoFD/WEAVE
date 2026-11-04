#!/usr/bin/env python3
"""把 sweep 目录的 metrics.csv + dino_metrics.csv 汇总成一张表：name n dino_s dino_s_x dino_c clip_s lpips。"""
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    rows_out = []
    for d in sorted(root.iterdir()):
        if not (d / "metrics.csv").exists():
            continue
        rows = list(csv.DictReader((d / "metrics.csv").open()))
        dino = list(csv.DictReader((d / "dino_metrics.csv").open()))
        for r, x in zip(rows, dino):
            r["ds"] = float(x["dino_style_similarity"])
            r["dc"] = float(x["dino_content_similarity"])
            r["cs"] = float(r["clip_style"])
            r["lp"] = float(r["lpips"])
            r["cross"] = r["src_style"] != r["tgt_style"]
        def m(rs, k):
            return round(sum(r[k] for r in rs) / len(rs), 4) if rs else None
        cx = [r for r in rows if r["cross"]]
        rows_out.append({"name": d.name, "n": len(rows),
                         "dino_s": m(rows, "ds"), "dino_s_x": m(cx, "ds"),
                         "dino_c": m(rows, "dc"), "dino_c_x": m(cx, "dc"),
                         "clip_s": m(rows, "cs"), "clip_s_x": m(cx, "cs"),
                         "lpips": m(rows, "lp"), "lpips_x": m(cx, "lp")})
    print("name                  n   dino_s dino_s_x  dino_c dino_c_x  clip_s   lpips")
    for r in rows_out:
        print("%-20s %4d %7.4f %8.4f %7.4f %8.4f %7.4f %7.4f" % (
            r["name"], r["n"], r["dino_s"], r["dino_s_x"], r["dino_c"], r["dino_c_x"],
            r["clip_s"], r["lpips"]))
    (root / "table.json").write_text(json.dumps(rows_out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
