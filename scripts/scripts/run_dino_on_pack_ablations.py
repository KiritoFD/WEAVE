"""Run paper-canonical DINO on every ablation board that has images but no dino_metrics.csv."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "results/cross_protocol/submission_pack/boards/ablation"
TEST = ROOT / "data/test"
PY = sys.executable
SCRIPT = ROOT / "utils/compute_dino_metrics.py"


def ensure_metrics_stub(board: Path) -> Path | None:
    """compute_dino_metrics requires metrics.csv with src/tgt/gen columns."""
    m = board / "metrics.csv"
    if m.is_file():
        return m
    return None


def main():
    boards = []
    for d in sorted(ABL.iterdir()):
        if not d.is_dir():
            continue
        img = d / "images"
        if not img.is_dir():
            continue
        n = sum(1 for p in img.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if n < 100:
            print("skip small", d.name, n)
            continue
        if (d / "dino_metrics.csv").is_file():
            print("have dino_metrics", d.name)
            continue
        if not (d / "metrics.csv").is_file():
            print("skip no metrics.csv", d.name)
            continue
        boards.append(d)

    print(f"Will run DINO on {len(boards)} boards")
    for d in boards:
        print("\n====", d.name, "====")
        cmd = [
            PY,
            "-u",
            str(SCRIPT),
            "--eval_dir",
            str(d),
            "--test_dir",
            str(TEST),
            "--batch_size",
            "4",
            "--max_refs_per_style",
            "30",
            "--output_csv",
            str(d / "dino_metrics.csv"),
            "--output_json",
            str(d / "dino_summary_cross.json"),
        ]
        # metrics.csv may point gen_image without images/ prefix; compute_dino resolves both
        r = subprocess.run(cmd, cwd=str(ROOT))
        print("exit", r.returncode, d.name)
        if r.returncode != 0:
            # try with eval_dir pointing such that gen paths work: copy metrics gen paths
            print("FAILED", d.name)


if __name__ == "__main__":
    main()
