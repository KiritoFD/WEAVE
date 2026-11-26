"""CLEAVE V34: Analytic Starting Point (A), Continuous Style Conditioning (B),
Joint Training & 44M Capacity Expansion (C), and Independent Holdout Verification (D).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import exp.v29_biglittle as v29_pkg
from exp.v29_biglittle import (
    assert_vram_budget,
    atomic_save,
    compute_csls_k10,
    compute_dino_top5_mean,
    peak_vram_gb,
)
from exp.v33_semantic_control import (
    POOL_C_PATH,
    S085_C10_KW,
    STYLES,
    V32_ECORE_CKPT,
    V32_OUT,
    V32_PCORE_CKPT,
    RawWriterV33,
    compute_style_pixel_stats,
    paired_ci_vs_baseline,
)
from src.eval.runner import run_measure_subprocess
from src.utils.runtime import REPO as ROOT, read_json, save_json

OUT = ROOT / "runs/v34_frontier"
SEED_V34 = 20261134


class RawWriterV34(RawWriterV33):
    """RawWriter saving batches directly under runs/v34_frontier/raw/<arm>."""

    def __init__(self, arm_name: str, dec=None, device: str = "cuda", *, expected_tasks: int = 750):
        super().__init__(arm_name, dec=dec, device=device)
        self.expected_tasks = expected_tasks
        self.path = OUT / "raw" / arm_name
        self.path.mkdir(parents=True, exist_ok=True)


def _to_float(val: Any, key_hint: str = "") -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        if key_hint and key_hint in val and isinstance(val[key_hint], (int, float)):
            return float(val[key_hint])
        for k in ("cross600", "mean", "value", "top5_mean", "csls_k10"):
            if k in val and isinstance(val[k], (int, float)):
                return float(val[k])
    raise TypeError(f"Cannot coerce {type(val)} to float: {val}")


def enrich_metrics_v34(arm_name: str, meas_dir: Path | None = None) -> dict[str, Any]:
    m_dir = meas_dir or (OUT / "measured" / arm_name)
    s = read_json(m_dir / "summary_v34.json")
    old_out = v29_pkg.OUT
    v29_pkg.OUT = m_dir.parent.parent
    try:
        csls = _to_float(compute_csls_k10(m_dir.name), "csls_k10")
        top5 = _to_float(compute_dino_top5_mean(m_dir), "dino_s_top5_mean")
    finally:
        v29_pkg.OUT = old_out

    c = s["cross600"]
    sm = s["same150"]
    return {
        "arm": arm_name,
        "dino_c": round(float(c["dino_c"]), 4),
        "dino_s": round(float(c["dino_s"]), 4),
        "dino_s_top5": round(top5, 4),
        "csls_k10": round(csls, 4),
        "csd_top1": round(float(c["csd_top1"]), 4),
        "clip_s": round(float(c["clip_s"]), 4),
        "lpips": round(float(c["lpips"]), 4),
        "style_affinity": round(float(c["style_affinity"]), 4),
        "pred_eq_src": round(float(c["pred_eq_src"]), 4),
        "same150_top1": round(float(sm["csd_top1"]), 4),
        "same150_dino_c": round(float(sm["dino_c"]), 4),
        "per_tgt_dino_s": {k[:3]: round(float(v["dino_s"]), 4) for k, v in s.get("per_target_cross120", {}).items()},
        "per_tgt_dino_c": {k[:3]: round(float(v["dino_c"]), 4) for k, v in s.get("per_target_cross120", {}).items()},
        "per_tgt_csd_top1": {k[:3]: round(float(v["csd_top1"]), 4) for k, v in s.get("per_target_cross120", {}).items()},
    }


def measure_and_summarize_v34(arm_name: str, *, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    raw_dir = OUT / "raw" / arm_name
    m_dir = OUT / "measured" / arm_name
    if force or not (m_dir / "summary_v34.json").exists():
        summary = run_measure_subprocess(raw_dir, m_dir, arm_name=arm_name, device=device, force=force)
        save_json(m_dir / "summary_v34.json", summary)
    return enrich_metrics_v34(arm_name, m_dir)


def v32_frontier_points() -> list[tuple[float, float]]:
    ledger = read_json(V32_OUT / "v32_final_ledger.json")["spectrum"]
    pts = []
    for k in (
        "ECore_NoDINO_l075",
        "PCore_NoDINO_s075_cal_c10",
        "PCore_NoDINO_s085_c10",
        "PCore_NoDINO_s093_cal_c10",
        "PCore_NoDINO_s100",
        "PCore_NoDINO_s125",
        "PCore_NoDINO_s150",
    ):
        if k in ledger:
            pts.append((float(ledger[k]["dino_c"]), float(ledger[k]["dino_s"])))
    return sorted(pts, key=lambda x: x[0])


def interp_s_at_c(c_val: float, pts: list[tuple[float, float]]) -> float:
    if c_val <= pts[0][0]:
        return pts[0][1]
    if c_val >= pts[-1][0]:
        return pts[-1][1]
    for (c0, s0), (c1, s1) in zip(pts, pts[1:]):
        if c0 <= c_val <= c1:
            a = (c_val - c0) / max(c1 - c0, 1e-6)
            return (1.0 - a) * s0 + a * s1
    return pts[-1][1]


__all__ = [
    "OUT",
    "POOL_C_PATH",
    "ROOT",
    "RawWriterV34",
    "S085_C10_KW",
    "SEED_V34",
    "STYLES",
    "V32_ECORE_CKPT",
    "V32_OUT",
    "V32_PCORE_CKPT",
    "assert_vram_budget",
    "atomic_save",
    "compute_style_pixel_stats",
    "enrich_metrics_v34",
    "interp_s_at_c",
    "measure_and_summarize_v34",
    "paired_ci_vs_baseline",
    "peak_vram_gb",
    "v32_frontier_points",
]
