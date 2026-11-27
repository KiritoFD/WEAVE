"""CLEAVE V35 Production Inference, Profiling, TRT Export & Deployment Package (RTX 3060 / 4090)."""
from __future__ import annotations

from pathlib import Path

from src.utils.runtime import REPO as ROOT

BUNDLE_DEFAULT_DIR = ROOT / "runs/deploy_bundle_v35"
OPT_OUT_DIR = ROOT / "runs/v35_3060_opt"

P8_LOCKED_CFG = {
    "pcore_ckpt": "runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt",
    "pcore_weight": "ema",
    "pcore_step": 3500,
    "ecore_ckpt": "runs/v32_nodino/ecore/ECore_NoDINO/ECore_NoDINO_final.pt",
    "a0_ckpt": "runs/a0/a0_encoder_only.pt",
    "s10b_ckpt": "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
    "k8_bank": "runs/v32_nodino/cache/k8_cnx417_bank.pt",
    "a3_priors": "runs/v34_frontier/cache/v34_analytic_priors.pt",
    "tiny_tensors": "runs/v35_final/cache/v35_tiny_tensors.pt",
    "ws_mul": 1.0,
    "wc_mul": 1.2,
    "dtau0": -0.05,
    "steps": 6,
    "alpha_y": 1.0,
    "alpha_c": 1.0,
    "scaling_factor": 0.18215,
}

E_BENCH_KNOBS = (0.25, 0.50, 0.60, 0.90)
E_GATE_KNOBS = (0.50, 0.60)
P_BENCH_KNOBS = (0.60, 0.85, 1.00, 1.50)

NON_INFERIORITY_LIMITS = {
    "dino_c_min_delta": -0.005,
    "dino_s_min_delta": -0.005,
    "csd_top1_pp_min_delta": -2.0,
    "same150_pp_min_delta": -2.0,
    "lpips_max_delta": 0.005,
}
