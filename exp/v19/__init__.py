"""PRISM v2 / V19: Orthogonal Quotient-Fiber Duality, Anisotropic S10-VGG Teacher & Source-Label-Free 1-Step Chord Student."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from exp.v18 import (
    EVAL_SEED,
    ROOT,
    STYLES,
    RawWriterV18,
    autocast,
    canonical_pid,
    clean_balanced_pool,
    read_json,
    save_json,
    seed_all,
    summarize_measured_arm,
    tasks_and_sources,
    utc,
)

OUT = ROOT / "runs/v19"
V18_OUT = ROOT / "runs/v18"


class RawWriterV19(RawWriterV18):
    def __init__(self, arm_name: str):
        super().__init__(arm_name, raw_root=OUT / "raw")


def run_measure_v19(arm_name: str, device: str = "cuda", save_png: bool = False) -> dict:
    """Run frozen 4-backbone referee in a strictly isolated child process."""
    raw_dir = OUT / "raw" / arm_name
    meas_dir = OUT / "measured" / arm_name
    if not (meas_dir / "eval.json").exists():
        cmd = [
            sys.executable, "-u", "-m", "src.eval.measure",
            "--raw", str(raw_dir),
            "--out", str(meas_dir),
            "--device", device,
        ]
        if not save_png:
            cmd.append("--no-png")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise RuntimeError(f"src.eval.measure failed for {arm_name} (rc={r.returncode})")
    s = summarize_measured_arm(arm_name, meas_dir)
    save_json(meas_dir / "summary_v19.json", s)
    c = s["cross600"]
    pt = {k[:3]: round(v["csd_top1"] * 100, 1) for k, v in s["per_target_cross120"].items()}
    print(
        f"[V19 EVAL] {arm_name:28s} | Cross600 Top1={c['csd_top1']*100:5.2f}% "
        f"Mar={c['csd_margin']:+.4f} DINO-S={c['dino_s']:.4f} Affinity={c['style_affinity']:.4f} "
        f"DINO-C={c['dino_c']:.4f} LPIPS={c['lpips']:.4f} SrcLock={c['pred_eq_src']*100:4.1f}% "
        f"SameTop1={s['same150']['csd_top1']*100:4.1f}% | {pt}",
        flush=True,
    )
    return s
