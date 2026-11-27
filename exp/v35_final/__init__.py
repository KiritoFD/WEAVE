"""CLEAVE V35-Finalization (12h Plan: P0-P8 Attribution, Reproducibility, & Release).

Modules:
  - common.py: shared loaders, parameterized test750/calib/holdout evaluators, E2E profiler,
    aligned content-representation builders (Stiefel / Whiten-32 / PCA-32), and unified 44M trainer.
  - p0_p2_evals.py: P0 baseline & integrity audit, P1 E-Core 7-arm causal ablation, P2 D0/D1/D2 decoder ablation.
  - p3_p4_retrain.py: P3 Stiefel & Pool C joint ablation + probes + 6-arm intervention, P4 Seed-2 verification.
  - p5_p8_finalize.py: P5 duration/EMA curve, P6 hyper calibration & invariants, P7 E/P necessity, P8 final ledger.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from exp.v29_biglittle import assert_vram_budget, atomic_save, peak_vram_gb
from exp.v33_semantic_control import (
    POOL_C_PATH,
    STYLES,
    V32_ECORE_CKPT,
    V32_OUT,
    V32_PCORE_CKPT,
    RawWriterV33,
    compute_style_pixel_stats,
    paired_ci_vs_baseline,
)
from exp.v34_frontier import (
    enrich_metrics_v34,
    interp_s_at_c,
    v32_frontier_points,
)
from exp.v34_frontier.stage_a_startpoint import (
    build_or_load_analytic_priors,
    decompose_py_luma,
)
from src.eval.runner import run_measure_batch_subprocess, run_measure_subprocess
from src.utils import S10_DECODER, auto_decode_batch, decode_s10, load_s10_decoder
from src.utils.runtime import REPO as ROOT, save_json

OUT = ROOT / "runs/v35_final"
SEED_V35 = 20261135
C2_V34_CKPT = ROOT / "runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt"
S10B_CKPT = S10_DECODER
D0_FULL_VAE_DIR = ROOT / "runs/v26/cache/sd0_vae_full"
D1_FULL_FT_CKPT = ROOT / "runs/v27/icme_dec/D1_full_ft.pt"


class RawWriterV35(RawWriterV33):
    """RawWriter saving batches directly under runs/v35_final/raw/<arm> with hardware-scaled S10 decoding."""

    def __init__(self, arm_name: str, dec=None, device: str = "cuda", *, raw_root: Path | None = None, expected_tasks: int = 750):
        super().__init__(arm_name, dec=dec, device=device)
        self.expected_tasks = int(expected_tasks)
        self.path = (raw_root or (OUT / "raw")) / arm_name
        self.path.mkdir(parents=True, exist_ok=True)
        self._decode_batch = auto_decode_batch(device)

    def write_batch(self, z_final: torch.Tensor, tasks_slice: list[dict], *, rgb_override: torch.Tensor | None = None) -> None:
        if rgb_override is not None:
            ids = [t["task_id"] for t in tasks_slice]
            self.append(ids, rgb_override.clamp(0.0, 1.0), z_final=z_final)
            return
        if self._dec is None:
            self._dec = load_s10_decoder(self._device)
        ids = [t["task_id"] for t in tasks_slice]
        rgb01 = decode_s10(self._dec, z_final, batch=self._decode_batch).clamp(0.0, 1.0)
        self.append(ids, rgb01, z_final=z_final)


def build_pcore_startpoint(
    z_e: torch.Tensor,
    tid: torch.Tensor,
    priors_a: dict[str, torch.Tensor],
    *,
    alpha_y: float = 1.0,
    alpha_c: float = 1.0,
    kernel_size: int = 19,
    sigma: float = 4.5,
) -> torch.Tensor:
    """Inlined P-Core startpoint constructor (keeps E-Core.forward untouched):

        z'_E = z_E + alpha_Y * Delta_L_low + alpha_C * Delta_P_C

    Zero learnable parameters, zero per-style branching, generalizes A0..A5 and P1 A1 raw chroma alignment.
    """
    ay = float(alpha_y)
    ac = float(alpha_c)
    if ay == 0.0 and ac == 0.0:
        return z_e

    dev = z_e.device
    zf = z_e.float()
    p_y = priors_a["P_Y"].to(dev)
    p_c = priors_a["P_C"].to(dev)
    u_y = priors_a["u_y"].to(dev)

    delta = torch.zeros_like(zf)
    if ay != 0.0:
        l_low, _, _ = decompose_py_luma(zf, p_y, kernel_size=kernel_size, sigma=sigma)
        s_low = torch.einsum("c,bchw->bhw", u_y, l_low).unsqueeze(1)
        mu_low_t = priors_a["mu_low_1d"].to(dev)[tid]
        std_low_t = priors_a["std_low_1d"].to(dev)[tid]
        mu_s = s_low.mean(dim=(2, 3), keepdim=True)
        std_s = s_low.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        s_low_new = (s_low - mu_s) / std_s * std_low_t + mu_low_t
        l_low_new = s_low_new * u_y.view(1, 4, 1, 1)
        delta = delta + ay * (l_low_new - l_low)

    if ac != 0.0:
        zc = torch.einsum("ij,bjhw->bihw", p_c, zf)
        mu_c_t = priors_a["mu_c"].to(dev)[tid]
        std_c_t = priors_a["std_c"].to(dev)[tid]
        mu_cs = zc.mean(dim=(2, 3), keepdim=True)
        std_cs = zc.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        zc_new = (zc - mu_cs) / std_cs * std_c_t + mu_c_t
        zc_matched = torch.einsum("ij,bjhw->bihw", p_c, zc_new)
        delta = delta + ac * (zc_matched - zc)

    return zf + delta


def measure_and_summarize_batch_v35(arms: list[str], *, device: str = "cuda", force: bool = False) -> dict[str, dict[str, Any]]:
    """Measure multiple test750 arms in ONE isolated FeatureStack subprocess and enrich metrics."""
    pending_items: list[tuple[Path, Path, str]] = []
    for arm in arms:
        m_dir = OUT / "measured" / arm
        if force or not (m_dir / "summary_v34.json").exists():
            pending_items.append((OUT / "raw" / arm, m_dir, arm))
    if pending_items:
        summaries = run_measure_batch_subprocess(pending_items, device=device, force=force)
        for arm, summary in summaries.items():
            save_json(OUT / "measured" / arm / "summary_v34.json", summary)
    return {arm: enrich_metrics_v34(arm, OUT / "measured" / arm) for arm in arms}


def measure_and_summarize_v35(arm_name: str, *, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    return measure_and_summarize_batch_v35([arm_name], device=device, force=force)[arm_name]


def summarize_matched_c(levels: dict[str, dict[str, Any]], c_grid: tuple[float, ...] = (0.50, 0.61, 0.70)) -> dict[str, Any]:
    """Interpolate DINO-S and CSD Top-1 at fixed DINO-C targets with explicit bracketing verification."""
    v32_pts = v32_frontier_points()
    pts_s = sorted([(float(m["dino_c"]), float(m["dino_s"])) for k, m in levels.items() if "wrong" not in k], key=lambda x: x[0])
    pts_t = sorted([(float(m["dino_c"]), float(m["csd_top1"])) for k, m in levels.items() if "wrong" not in k], key=lambda x: x[0])
    c_lo, c_hi = pts_s[0][0], pts_s[-1][0]
    bracketed = [bool(c_lo - 1e-6 <= c <= c_hi + 1e-6) for c in c_grid]
    s_at_c = [round(interp_s_at_c(c, pts_s), 4) for c in c_grid]
    t_at_c = [round(interp_s_at_c(c, pts_t), 4) for c in c_grid]
    ds_v32 = [round(s_at_c[i] - interp_s_at_c(c_grid[i], v32_pts), 4) for i in range(len(c_grid))]
    return {
        "c_grid": list(c_grid),
        "c_range": [round(c_lo, 4), round(c_hi, 4)],
        "bracketed": bracketed,
        "all_bracketed": all(bracketed),
        "dino_s_at_c": s_at_c,
        "csd_top1_at_c": t_at_c,
        "delta_s_vs_v32": ds_v32,
        "mean_delta_s_vs_v32": round(sum(ds_v32) / len(ds_v32), 4),
    }


__all__ = [
    "C2_V34_CKPT",
    "D0_FULL_VAE_DIR",
    "D1_FULL_FT_CKPT",
    "OUT",
    "POOL_C_PATH",
    "ROOT",
    "RawWriterV35",
    "S10B_CKPT",
    "SEED_V35",
    "STYLES",
    "V32_ECORE_CKPT",
    "V32_OUT",
    "V32_PCORE_CKPT",
    "assert_vram_budget",
    "atomic_save",
    "build_or_load_analytic_priors",
    "build_pcore_startpoint",
    "compute_style_pixel_stats",
    "measure_and_summarize_batch_v35",
    "measure_and_summarize_v35",
    "paired_ci_vs_baseline",
    "peak_vram_gb",
    "summarize_matched_c",
]
