#!/usr/bin/env python3
"""Build the frozen 750-task manifest for image-guided baselines (StyleGallery,
Attention Distillation, ...).

For every task in eval_protocol/tasks_750.json, attaches ONE fixed reference
image per TARGET style drawn from the TRAIN pool only (data/train/<style>/),
selected with a fixed seed — never from data/test or the eval reference set.

Outputs:
  --out       CSV: task_id,source_path,style_paths,source_style,target_style,is_cross
  --refs-out  JSON: chosen reference path per style (provenance)
"""
import argparse
import csv
import json
import random
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weave-root", required=True, type=Path)
    ap.add_argument("--tasks", default="eval_protocol/tasks_750.json")
    ap.add_argument("--ref-root", default="data/train")
    ap.add_argument("--refs-per-style", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--refs-out", required=True, type=Path)
    args = ap.parse_args()

    root = args.weave_root.resolve()
    tasks_path = root / args.tasks
    data = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = data["tasks"] if isinstance(data, dict) else data
    styles = data.get("styles") if isinstance(data, dict) else None

    rng = random.Random(args.seed)
    refs = {}
    for style in (styles or sorted({t["target_style"] for t in tasks})):
        pool_dir = root / args.ref_root / style
        files = sorted(p for p in pool_dir.iterdir()
                       if p.suffix.lower() in IMG_EXT) if pool_dir.is_dir() else []
        if not files:
            raise FileNotFoundError(f"no train references under {pool_dir}")
        refs[style] = [str(p) for p in
                       rng.sample(files, min(args.refs_per_style, len(files)))]

    args.refs_out.parent.mkdir(parents=True, exist_ok=True)
    args.refs_out.write_text(json.dumps(
        {"seed": args.seed, "ref_root": args.ref_root,
         "refs_per_style": args.refs_per_style, "styles": refs},
        indent=1), encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["task_id", "source_path", "style_paths",
                    "source_style", "target_style", "is_cross"])
        for t in tasks:
            src = root / t["source_rel_path"]
            if not src.is_file():
                raise FileNotFoundError(f"missing source {src}")
            style_paths = ";".join(refs[t["target_style"]])
            w.writerow([t["task_id"], str(src), style_paths,
                        t["source_style"], t["target_style"],
                        int(t["is_cross"])])
            n += 1
    print(f"wrote {n} tasks -> {args.out}; refs -> {args.refs_out}")


if __name__ == "__main__":
    main()
