"""CLEAVE V33: Quality Frontier Breakthrough Suite (P0 Routing/Luma -> P1 Q64+ -> P2 Semantic Control -> P3 Joint Verify)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import exp.v29_biglittle as v29_pkg
from exp.v29_biglittle import (
    RawWriterV29,
    assert_vram_budget,
    atomic_save,
    compute_csls_k10,
    compute_dino_top5_mean,
    peak_vram_gb,
)
from src.eval.metrics import paired_cluster_ci
from src.eval.runner import run_measure_subprocess
from src.utils import decode_s10, load_s10_decoder
from src.utils.runtime import REPO as ROOT, read_json, save_json
from src.utils.structure import dwt2_haar, rgb_to_gray, rgb_to_oklab

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
OUT = ROOT / "runs/v33_semantic_control"
SEED_V33 = 20261133

V32_OUT = ROOT / "runs/v32_nodino"
V32_ECORE_CKPT = V32_OUT / "ecore/ECore_NoDINO/ECore_NoDINO_final.pt"
V32_PCORE_CKPT = V32_OUT / "pcore/PCore_NoDINO/PCore_NoDINO_s3500.pt"
POOL_C_PATH = ROOT / "runs/v28_unified/pools/pool_C.pt"

# V32 calibrated s=0.85_c10 working point (unique V32 production baseline)
S085_C10_KW: dict[str, float] = {
    "draft_lambda": 0.98,
    "tau0": 0.790,
    "ws": 4.00,
    "wc": 2.60,
    "wsp": 2.00,
    "rescale_phi": 0.0,
    "gamma": 1.0,
    "lscale": 0.92,
}


class RawWriterV33(RawWriterV29):
    """RawWriter saving batches directly under runs/v33_semantic_control/raw/<arm>."""

    def __init__(self, arm_name: str, dec=None, device: str = "cuda"):
        super().__init__(arm_name, dec=dec, device=device)
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


def enrich_metrics_v33(arm_name: str, meas_dir: Path | None = None) -> dict[str, Any]:
    """Read summary_v33.json and enrich with DINO-S Top-5 Mean, CSLS k=10, and source-leakage rate."""
    m_dir = meas_dir or (OUT / "measured" / arm_name)
    s = read_json(m_dir / "summary_v33.json")
    old_out = v29_pkg.OUT
    v29_pkg.OUT = m_dir.parent.parent
    try:
        csls_d = compute_csls_k10(m_dir.name)
        csls = _to_float(csls_d, "csls_k10")
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


def measure_and_summarize(arm_name: str, *, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    """Run process-isolated referee src.eval.measure on raw/<arm_name> and return enriched summary."""
    raw_dir = OUT / "raw" / arm_name
    m_dir = OUT / "measured" / arm_name
    if force or not (m_dir / "summary_v33.json").exists():
        summary = run_measure_subprocess(raw_dir, m_dir, arm_name=arm_name, device=device, force=force)
        save_json(m_dir / "summary_v33.json", summary)
    return enrich_metrics_v33(arm_name, m_dir)


def texture_energy_rgb(rgb01: torch.Tensor) -> torch.Tensor:
    """Continuous style-agnostic texture & edge energy E(x) from Haar HF + luminance gradient."""
    y = rgb_to_gray(rgb01.float().clamp(0.0, 1.0))
    dy = (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    dx = (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    _, lh, hl, hh = dwt2_haar(y)
    hf = (lh.abs() + hl.abs() + hh.abs()).mean(dim=(1, 2, 3)) / 3.0
    return 0.5 * (dx + dy) + hf


@torch.no_grad()
def compute_style_pixel_stats(raw_dir: Path, target_style_id: int = 2, cross_only: bool = True) -> dict[str, float]:
    """Compute HSV saturation, Oklab chroma, gray_std, and TV_grad over raw output batches for one target style."""
    meta = read_json(raw_dir / "meta.json")
    tasks = {t["task_id"]: t for t in read_json(ROOT / "eval_protocol/tasks_750.json")["tasks"]}
    sats, chromas, gstds, tvs = [], [], [], []
    for entry in meta["batches"]:
        pl = torch.load(raw_dir / entry["file"], map_location="cpu", weights_only=False)
        ids = pl["ids"]
        if "rgb_u8" in pl:
            imgs = pl["rgb_u8"].float() / 255.0
        elif "rgb01" in pl:
            imgs = pl["rgb01"].float()
        else:
            imgs = pl["images"].float()
            if imgs.max() > 1.5:
                imgs = imgs / 255.0
        keep = []
        for idx, tid_s in enumerate(ids):
            t = tasks[tid_s]
            if int(t["target_style_id"]) == target_style_id and (not cross_only or int(t["source_style_id"]) != target_style_id):
                keep.append(idx)
        if not keep:
            continue
        sub = imgs[keep].clamp(0.0, 1.0)
        mx = sub.max(dim=1).values
        mn = sub.min(dim=1).values
        sat = ((mx - mn) / mx.clamp_min(1e-6)).mean(dim=(1, 2))
        lab = rgb_to_oklab(sub)
        chroma = (lab[:, 1].square() + lab[:, 2].square() + 1e-12).sqrt().mean(dim=(1, 2))
        y = rgb_to_gray(sub)
        gstd = y.flatten(1).std(dim=1, unbiased=False)
        tv = 0.5 * ((y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean(dim=(1, 2, 3)) + (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean(dim=(1, 2, 3)))
        sats.extend(sat.tolist())
        chromas.extend(chroma.tolist())
        gstds.extend(gstd.tolist())
        tvs.extend(tv.tolist())
    n = max(1, len(sats))
    return {
        "n": len(sats),
        "sat_hsv": round(sum(sats) / n, 4),
        "chroma_oklab": round(sum(chromas) / n, 4),
        "gray_std": round(sum(gstds) / n, 4),
        "tv_grad": round(sum(tvs) / n, 4),
    }


def paired_ci_vs_baseline(cand_eval_json: Path, base_eval_json: Path, *, cross_only: bool = True) -> dict[str, Any]:
    """150-source cluster paired Bootstrap 95% CI between candidate and baseline eval.json."""
    rows_a = read_json(cand_eval_json)["rows"]
    rows_b = read_json(base_eval_json)["rows"]
    ci_all = paired_cluster_ci(
        rows_a,
        rows_b,
        metrics=("dino_c", "dino_s", "lpips", "target_top1", "clip_s"),
        cross_only=cross_only,
        n_boot=4000,
        seed=SEED_V33,
    )
    map_a = {r["task_id"]: r for r in rows_a if (not cross_only or r["is_cross"])}
    map_b = {r["task_id"]: r for r in rows_b if (not cross_only or r["is_cross"])}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    out: dict[str, Any] = {"n_tasks": len(common)}
    for metric in ("dino_c", "dino_s", "lpips", "target_top1", "clip_s"):
        va = float(np.mean([float(map_a[k][metric if metric != "clip_s" else "clip_scores_5"][map_a[k]["tgt_style_id"]] if metric == "clip_s" else map_a[k][metric]) for k in common]))
        vb = float(np.mean([float(map_b[k][metric if metric != "clip_s" else "clip_scores_5"][map_b[k]["tgt_style_id"]] if metric == "clip_s" else map_b[k][metric]) for k in common]))
        ci = ci_all[metric]
        out[metric] = {
            "cand": round(va, 4),
            "base": round(vb, 4),
            "diff": round(float(ci["mean"]), 4),
            "ci95": [round(float(ci["ci95"][0]), 4), round(float(ci["ci95"][1]), 4)],
            "excludes_zero": bool(ci["excludes_zero"]),
        }
    return out


__all__ = [
    "OUT",
    "POOL_C_PATH",
    "ROOT",
    "RawWriterV33",
    "S085_C10_KW",
    "SEED_V33",
    "STYLES",
    "V32_ECORE_CKPT",
    "V32_OUT",
    "V32_PCORE_CKPT",
    "assert_vram_budget",
    "atomic_save",
    "compute_style_pixel_stats",
    "enrich_metrics_v33",
    "measure_and_summarize",
    "paired_ci_vs_baseline",
    "peak_vram_gb",
    "texture_energy_rgb",
]
