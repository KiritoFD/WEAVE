#!/usr/bin/env python3
"""Strict sequential, resumable AD batch for the frozen 750-task manifest.

One fresh subprocess per task (VRAM isolation on 12GB). cold_wall_s includes
model load — NOT comparable to warm inference. Failure stops the batch.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", default="ad750_outputs", type=Path)
    ap.add_argument("--runner", required=True, type=Path,
                    help="path to run_ad_smoke_12g.py")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--content-weight", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if args.limit:
        rows = rows[:args.limit]

    logpath = out / "execution.jsonl"
    for i, row in enumerate(rows):
        if i < args.start:
            continue
        tid = row["task_id"]
        safe = f"{i:04d}_" + hashlib.sha256(tid.encode()).hexdigest()[:10]
        result = out / f"{safe}.png"
        if result.is_file():
            print(f"[SKIP] {i+1}/{len(rows)} {tid}")
            continue
        src = Path(row["source_path"])
        refs = row["style_paths"].split(";")
        if not src.is_file():
            raise FileNotFoundError(src)
        if not refs or not Path(refs[0]).is_file():
            raise FileNotFoundError(f"bad refs for {tid}: {refs}")

        taskdir = out / "_work" / safe
        taskdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(args.runner.resolve()),
               "--repo", str(args.repo.resolve()),
               "--content", str(src), "--style", refs[0],
               "--steps", str(args.steps), "--precision", args.precision,
               "--content-weight", str(args.content_weight),
               "--lr", str(args.lr), "--out", str(taskdir)]
        lp = out / f"{safe}.log"
        t0 = time.monotonic()
        with lp.open("w", encoding="utf-8") as lfh:
            p = subprocess.run(cmd, stdout=lfh, stderr=subprocess.STDOUT)
        dt = time.monotonic() - t0

        produced = sorted(taskdir.rglob("*.png"))
        success = p.returncode == 0 and produced
        if success:
            shutil.copyfile(produced[-1], result)
        meta = {"index": i, "task_id": tid,
                "source_style": row["source_style"],
                "target_style": row["target_style"],
                "is_cross": int(row["is_cross"]),
                "source_path": str(src), "ref_paths": refs,
                "result": str(result) if success else None,
                "returncode": p.returncode, "success": bool(success),
                "cold_wall_s": round(dt, 3), "steps": args.steps,
                "precision": args.precision}
        with logpath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"[{'OK' if success else 'FAIL'}] {i+1}/{len(rows)} {tid} {dt:.1f}s")
        if not success:
            raise RuntimeError(f"AD failed on {tid}; see {lp}. STOP.")


if __name__ == "__main__":
    main()
