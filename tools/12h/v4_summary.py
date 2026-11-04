#!/usr/bin/env python3
"""汇总 V4 评测目录：读 metrics.csv + dino_metrics.csv，给 600 跨风格 / 150 同风格拆分。

用法: python3 tools/12h/v4_summary.py <out_dir> [out_dir ...]
"""
import csv
import json
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with (path / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (path / "dino_metrics.csv").open(newline="", encoding="utf-8") as f:
        dino = list(csv.DictReader(f))
    if len(rows) != len(dino):
        raise ValueError(f"{path}: metrics/dino 行数不一致 {len(rows)} vs {len(dino)}")
    for r, d in zip(rows, dino):
        r.update({k: d[k] for k in ("dino_style_similarity", "dino_content_similarity",
                                    "dino_structure_mse")})
        r["same_style"] = r["src_style"] == r["tgt_style"]
    return rows


def mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return round(sum(vals) / len(vals), 5) if vals else None


def summarize(path: Path) -> dict:
    rows = load_rows(path)
    cross = [r for r in rows if not r["same_style"]]
    same = [r for r in rows if r["same_style"]]
    out = {"dir": path.name, "n": len(rows), "n_cross": len(cross), "n_same": len(same)}
    for key, name in (("dino_style_similarity", "dino_s"),
                      ("dino_content_similarity", "dino_c"),
                      ("clip_style", "clip_s"), ("lpips", "lpips")):
        out[name] = mean(rows, key)
        out[name + "_x"] = mean(cross, key)
        out[name + "_s"] = mean(same, key)
    meta = path / "v4_run_meta.txt"
    if meta.exists():
        out["meta"] = meta.read_text(encoding="utf-8").strip().replace("\n", " ")
    summ = path / "dino_summary.json"
    if summ.exists():
        out["dino_summary"] = json.loads(summ.read_text(encoding="utf-8"))
    return out


def per_style(path: Path, key_field: str) -> None:
    rows = load_rows(path)
    names = sorted({r[key_field] for r in rows})
    print(f"{path.name}  ({key_field})")
    print("%-20s %4s %7s %7s %7s %7s" % ("style", "n", "dino_s", "dino_c", "clip_s", "lpips"))
    for name in names:
        sel = [r for r in rows if r[key_field] == name]
        print("%-20s %4d %7.4f %7.4f %7.4f %7.4f" % (
            name, len(sel), mean(sel, "dino_style_similarity"),
            mean(sel, "dino_content_similarity"), mean(sel, "clip_style"), mean(sel, "lpips")))


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--per-style":
        for d in args[1:]:
            per_style(Path(d), "tgt_style")
            print()
            per_style(Path(d), "src_style")
            print()
        return
    dirs = [Path(p) for p in args]
    print(json.dumps([summarize(d) for d in dirs], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

