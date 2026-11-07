"""Fetch V5 JSON/Markdown artifacts from 3060 into the local WEAVE tree."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE_BASE = "/mnt/i/Latent_Style/WEAVE/"
OUT_BASE = ROOT / "runs" / "v5"

FILES = [
    "runs/v5/p0_report.json",
    "runs/v5/v5_eval200_tasks.json",
    "runs/v5/v5_oracle100_tasks.json",
    "runs/v5/v5_eval200_baseline.json",
    "runs/v5/p1_oracle/oracle_matrix_report.json",
    "runs/v5/b1_style_encoder/b1_report.json",
    "runs/v5/route_c/noise_check.json",
    "runs/v5/route_c/noise_check_lowmid.json",
    "runs/v5/route_c/quick_report.json",
    "runs/v5/route_c/quick_report_lowmid.json",
    "runs/v5/route_c/best_config.json",
    "runs/v5/route_c/final_report.json",
    "runs/v5/PROGRESS.md",
    "runs/v5/route_c/logs/train_full.log",
    "runs/v5/route_c/logs/train_lowmid.log",
    "runs/v5/route_c/logs/check.log",
    "runs/v5/route_c/logs/check_lowmid.log",
    "runs/v5/route_c/logs/quick.log",
    "runs/v5/route_c/logs/final.log",
]


def main():
    ok, skipped = 0, 0
    for rel in FILES:
        dst = ROOT / rel
        r = subprocess.run(
            ["ssh", "3060", "wsl", "cat", REMOTE_BASE + rel],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0 or not r.stdout:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(r.stdout, encoding="utf-8")
        ok += 1
        print(f"OK {rel} ({len(r.stdout)}B)")
    print(f"fetched={ok} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
