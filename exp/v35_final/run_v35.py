"""CLEAVE V35 Unified 12h Experiment Runner (P0-P8).

Usage:
  python -u -m exp.v35_final.run_v35 --stage smoke
  python -u -m exp.v35_final.run_v35 --stage all
"""
from __future__ import annotations

import argparse
from pathlib import Path

from exp.v35_final import OUT
from exp.v35_final.common import train_44m_ablation
from exp.v35_final.p0_p2_evals import run_p0, run_p1, run_p2
from exp.v35_final.p3_p4_retrain import run_p3, run_p4
from exp.v35_final.p5_p8_finalize import run_p5, run_p6, run_p7, run_p8
from exp.v35_final.p9_paper_audit import run_p9
from src.utils.runtime import save_json, utc


def run_smoke(device: str = "cuda") -> dict:
    """Fast 5-step smoke test on 44M ablation trainer verifying VRAM saturation and zero-NaN gradients."""
    tr = train_44m_ablation(
        "V35_smoke_whiten32",
        OUT / "smoke/whiten32",
        content_mode="whiten32",
        pool_c_mode="0pct_self_match",
        updates=3500,
        smoke_steps=5,
        device=device,
    )
    rep = {"updated_utc": utc(), "smoke": tr}
    save_json(OUT / "v35_smoke_report.json", rep)
    print(f"[V35 Smoke Done] Peak VRAM={tr['peak_vram_gb']:.2f}GB elapsed={tr['elapsed_min']}m", flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["smoke", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p0_p2", "all"],
        required=True,
    )
    ap.add_argument("--only", default=None, help="Optional sub-stage selector for p9 (e.g., p9a..p9e)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.stage == "smoke":
        run_smoke(device=args.device)
        return
    if args.stage == "p9":
        run_p9(device=args.device, only=args.only)
        return

    stages = {
        "p0": [run_p0],
        "p1": [run_p1],
        "p2": [run_p2],
        "p3": [run_p3],
        "p4": [run_p4],
        "p5": [run_p5],
        "p6": [run_p6],
        "p7": [run_p7],
        "p8": [run_p8],
        "p0_p2": [run_p0, run_p1, run_p2],
        # Run zero-training diagnostics & already-trained P5 before long retraining P3/P4 -> P6 -> P7 -> P8
        "all": [run_p0, run_p1, run_p2, run_p5, run_p3, run_p4, run_p6, run_p7, run_p8],
    }[args.stage]

    for fn in stages:
        fn(device=args.device)


if __name__ == "__main__":
    main()
