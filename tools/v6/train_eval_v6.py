"""WEAVE V6 Stage 1 & 2: Unified Trainer & Evaluator for D1 (Direct) and F1 (C³-Flow).

Implements:
  1. In-domain real D5 style prototype moments, WCT covariance roots, and patch-SWD
     quantile grids from the 15,545 real training latents (zero CSD, zero test leak).
  2. Training for D1 (Direct Operator G_phi(c, s)) and F1 (C³-Flow v_theta(z_tau, tau, c, s))
     with normalized Haar band loss + coherent real-painting reconstruction +
     cross-style canonical-content invariance.
  3. Section IV Mandatory 3-Part Conditioning Validity Audit:
     (1) L_correct < L_wrong on held-out real paintings with paired 95% bootstrap CI
     (2) 5-way Target Style Selectivity on identical canonical content c
     (3) Content Condition Swap sensitivity under fixed Style-ID
  4. 200-task cross-style validation & 750-task (600 cross + 150 same) full evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths  # noqa: E402
import csd_model  # noqa: E402
from wavelet import dwt2_haar, idwt2_haar  # noqa: E402
from tools.eval_fast import (  # noqa: E402
    METRIC_HEADER, DINO_HEADER, MetricEngine, Timer, build_protocol, write_csv,
)
from tools.pseudo_gt.matching import encode_dino, load_dino_model  # noqa: E402
from utils.run_evaluation import _load_eval_image_tensor, to_lpips_input  # noqa: E402
from tools.v5.route_c import csd_prototypes, decode_trt, load_trt, load_vae  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer, spatial_self_sim_corr, gradient_dir_cos  # noqa: E402
from tools.v6.models_v6 import C3UNet  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
V5_ROOT = REPO_ROOT / "runs" / "v5"
V6_ROOT = REPO_ROOT / "runs" / "v6"
RENDER_ROOT = paths.asset_root() / "eval_assets" / "v6_main"


def boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> List[float]:
    if len(x) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def matrix_sqrt_invsqrt(cov: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric positive-definite 4x4 matrix square root and inverse square root."""
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=torch.float32)
    eigvals, eigvecs = torch.linalg.eigh(cov.float() + eps * eye)
    eigvals = eigvals.clamp(min=eps)
    s = eigvecs @ torch.diag_embed(eigvals.sqrt()) @ eigvecs.transpose(-1, -2)
    inv_s = eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2)
    return s, inv_s


def build_indomain_style_targets(
    latents_by_style: Dict[str, torch.Tensor],
    asset_root: Path,
    device: str = "cuda",
    top_k: int = 64,
) -> dict:
    """Build in-domain prototypical real-painting subband moments & patch banks per style.

    Uses ONLY D5 training pool latents + DINOv2 train-pool centroids (zero CSD, zero test).
    """
    cache_path = V6_ROOT / "cache" / f"indomain_style_targets_v2_k{top_k}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    dino_dir = asset_root / "pseudo_gt_24h" / "dino_224"
    cls_by_style = {}
    centroids = []
    for s in STYLES:
        d = torch.load(dino_dir / f"{s}.pt", map_location="cpu", weights_only=False)
        c = F.normalize(d["cls"].float(), dim=-1)
        cls_by_style[s] = c
        centroids.append(F.normalize(c.mean(dim=0, keepdim=True), dim=-1))
    cent_mat = torch.cat(centroids, dim=0)  # (5, 384)

    rng = np.random.default_rng(42)
    R_ll = rng.normal(size=(64, 64)).astype(np.float32)
    R_ll /= np.linalg.norm(R_ll, axis=1, keepdims=True)
    R_hf = rng.normal(size=(64, 192)).astype(np.float32)
    R_hf /= np.linalg.norm(R_hf, axis=1, keepdims=True)

    targets = {
        "R_ll": torch.from_numpy(R_ll),
        "R_hf": torch.from_numpy(R_hf),
        "styles": {},
    }

    for sid, s in enumerate(STYLES):
        z_all = latents_by_style[s].float()
        c_vecs = cls_by_style[s][100 : 100 + z_all.shape[0]]
        sims = c_vecs @ cent_mat.t()  # (N, 5)
        own_sim = sims[:, sid]
        other_mask = [i for i in range(len(STYLES)) if i != sid]
        margin = own_sim - sims[:, other_mask].max(dim=1).values
        k = min(top_k, z_all.shape[0])
        top_idx = margin.topk(k).indices
        z_core = z_all[top_idx]  # (K, 4, 64, 64)

        ll, lh, hl, hh = dwt2_haar(z_core)
        # Within-image spatial mean and std, then averaged over core real paintings
        ll_mu = ll.mean(dim=(2, 3)).mean(dim=0)  # (4,)
        ll_sd = ll.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4)
        lh_mu = lh.mean(dim=(2, 3)).mean(dim=0)
        lh_sd = lh.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4)
        hl_mu = hl.mean(dim=(2, 3)).mean(dim=0)
        hl_sd = hl.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4)
        hh_mu = hh.mean(dim=(2, 3)).mean(dim=0)
        hh_sd = hh.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4)

        # Within-image 4x4 covariance averaged over core paintings
        def avg_within_cov(band: torch.Tensor) -> torch.Tensor:
            B, C, H, W = band.shape
            flat = band.reshape(B, C, H * W)
            flat = flat - flat.mean(dim=2, keepdim=True)
            covs = torch.bmm(flat, flat.transpose(1, 2)) / max(H * W - 1, 1)
            return covs.mean(dim=0)

        ll_cov = avg_within_cov(ll)
        ll_cov_sqrt, ll_cov_invsqrt = matrix_sqrt_invsqrt(ll_cov)

        # 2-level Haar DWT on LL for coarse structure/color energy
        ll2, lh2, hl2, hh2 = dwt2_haar(ll)
        ll2_mu = ll2.mean(dim=(2, 3)).mean(dim=0)
        ll2_sd = ll2.std(dim=(2, 3)).mean(dim=0).clamp(min=1e-4)

        # Patch SWD quantile grids on 4x4 LL patches (64-d) and 4x4 HF patches (192-d)
        pat_ll = F.unfold(ll, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 64).numpy()
        proj_ll = np.sort(pat_ll @ R_ll.T, axis=0)
        q_idx = (np.linspace(0.0, 1.0, 128) * (proj_ll.shape[0] - 1)).astype(np.int64)
        qgrid_ll = torch.from_numpy(proj_ll[q_idx].astype(np.float32))

        hf_cat = torch.cat([lh, hl, hh], dim=1)  # (K, 12, 32, 32)
        pat_hf = F.unfold(hf_cat, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 192).numpy()
        proj_hf = np.sort(pat_hf @ R_hf.T, axis=0)
        q_idx_hf = (np.linspace(0.0, 1.0, 128) * (proj_hf.shape[0] - 1)).astype(np.int64)
        qgrid_hf = torch.from_numpy(proj_hf[q_idx_hf].astype(np.float32))

        targets["styles"][s] = {
            "top_indices": top_idx,
            "core_latents": z_core.half(),
            "ll_mu": ll_mu, "ll_sd": ll_sd, "ll_cov": ll_cov,
            "ll_cov_sqrt": ll_cov_sqrt, "ll_cov_invsqrt": ll_cov_invsqrt,
            "lh_mu": lh_mu, "lh_sd": lh_sd,
            "hl_mu": hl_mu, "hl_sd": hl_sd,
            "hh_mu": hh_mu, "hh_sd": hh_sd,
            "ll2_mu": ll2_mu, "ll2_sd": ll2_sd,
            "qgrid_ll": qgrid_ll,
            "qgrid_hf": qgrid_hf,
        }
        print(f"[Targets] {s}: core={k}, ll_sd={ll_sd.mean():.3f}, hf_sd={(lh_sd.mean()+hl_sd.mean())/2:.3f}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(targets, cache_path)
    return targets


def make_coherent_real_target(
    z_real: torch.Tensor,
    style_ids: torch.Tensor,
    style_targets: dict,
    coherence_beta: float = 0.65,
) -> torch.Tensor:
    """Anchor a real painting's global subband palette/energy toward its style's core
    real distribution while preserving 100% of the painting's spatial brushwork.
    """
    if coherence_beta <= 0.0:
        return z_real
    ll, lh, hl, hh = dwt2_haar(z_real.float())
    B, C, H, W = ll.shape
    ll_out, lh_out, hl_out, hh_out = torch.empty_like(ll), torch.empty_like(lh), torch.empty_like(hl), torch.empty_like(hh)

    for sid, s in enumerate(STYLES):
        mask = (style_ids == sid)
        if not mask.any():
            continue
        st = style_targets["styles"][s]
        dev = z_real.device

        # Within-image subband alignment on LL toward style s core mean & std
        ll_sub = ll[mask]
        m_sub = ll_sub.mean(dim=(2, 3), keepdim=True)
        sd_sub = ll_sub.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        ll_norm = (ll_sub - m_sub) / sd_sub

        mu_t = st["ll_mu"].to(dev).view(1, 4, 1, 1)
        sd_t = st["ll_sd"].to(dev).view(1, 4, 1, 1)
        ll_proto = ll_norm * sd_t + mu_t
        ll_blend = (1.0 - coherence_beta) * ll_sub + coherence_beta * ll_proto
        ll_out[mask] = ll_blend

        for b_sub, b_out, mu_k, sd_k in (
            (lh[mask], lh_out, "lh_mu", "lh_sd"),
            (hl[mask], hl_out, "hl_mu", "hl_sd"),
            (hh[mask], hh_out, "hh_mu", "hh_sd"),
        ):
            mb = b_sub.mean(dim=(2, 3), keepdim=True)
            sb = b_sub.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            bn = (b_sub - mb) / sb
            mt = st[mu_k].to(dev).view(1, 4, 1, 1)
            st_d = st[sd_k].to(dev).view(1, 4, 1, 1)
            b_proto = bn * st_d + mt
            b_out[mask] = (1.0 - coherence_beta) * b_sub + coherence_beta * b_proto

    return idwt2_haar(ll_out, lh_out, hl_out, hh_out).to(dtype=z_real.dtype)


def compute_style_distribution_loss(
    z_pred: torch.Tensor,
    tgt_ids: torch.Tensor,
    style_targets: dict,
    b1_encoder: nn.Module | None = None,
    b1_protos: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-domain multi-scale wavelet moment + covariance + patch-SWD + B1 encoder loss."""
    dev = z_pred.device
    ll, lh, hl, hh = dwt2_haar(z_pred.float())
    ll2, _, _, _ = dwt2_haar(ll)
    R_ll = style_targets["R_ll"].to(dev)
    R_hf = style_targets["R_hf"].to(dev)

    B = z_pred.shape[0]
    total = z_pred.new_tensor(0.0)

    for sid, s in enumerate(STYLES):
        mask = (tgt_ids == sid)
        if not mask.any():
            continue
        st = style_targets["styles"][s]
        ll_s, lh_s, hl_s, hh_s = ll[mask], lh[mask], hl[mask], hh[mask]
        ll2_s = ll2[mask]

        # 1. Per-image spatial mean & std matching
        l_mom = (
            F.mse_loss(ll_s.mean(dim=(2, 3)), st["ll_mu"].to(dev).unsqueeze(0).expand(ll_s.shape[0], -1))
            + F.mse_loss(ll_s.std(dim=(2, 3)), st["ll_sd"].to(dev).unsqueeze(0).expand(ll_s.shape[0], -1))
            + F.mse_loss(lh_s.std(dim=(2, 3)), st["lh_sd"].to(dev).unsqueeze(0).expand(lh_s.shape[0], -1))
            + F.mse_loss(hl_s.std(dim=(2, 3)), st["hl_sd"].to(dev).unsqueeze(0).expand(hl_s.shape[0], -1))
            + F.mse_loss(hh_s.std(dim=(2, 3)), st["hh_sd"].to(dev).unsqueeze(0).expand(hh_s.shape[0], -1))
            + F.mse_loss(ll2_s.std(dim=(2, 3)), st["ll2_sd"].to(dev).unsqueeze(0).expand(ll2_s.shape[0], -1))
        )

        # 2. Within-image 4x4 LL covariance matching
        flat = ll_s.reshape(ll_s.shape[0], 4, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / max(flat.shape[2] - 1, 1)
        l_cov = F.mse_loss(cov.mean(dim=0), st["ll_cov"].to(dev))

        # 3. Multi-scale Patch SWD on LL and HF (stride=4 -> 64 patches/img, fast & low-VRAM)
        pat_ll = F.unfold(ll_s, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 64)
        proj_ll, _ = torch.sort(pat_ll @ R_ll.t(), dim=0)
        q_idx = torch.linspace(0, proj_ll.shape[0] - 1, 128, device=dev).long()
        l_swd_ll = F.l1_loss(proj_ll[q_idx], st["qgrid_ll"].to(dev))

        hf_cat = torch.cat([lh_s, hl_s, hh_s], dim=1)
        pat_hf = F.unfold(hf_cat, kernel_size=4, stride=4).permute(0, 2, 1).reshape(-1, 192)
        proj_hf, _ = torch.sort(pat_hf @ R_hf.t(), dim=0)
        q_idx_h = torch.linspace(0, proj_hf.shape[0] - 1, 128, device=dev).long()
        l_swd_hf = F.l1_loss(proj_hf[q_idx_h], st["qgrid_hf"].to(dev))

        sub = l_mom + 0.5 * l_cov + 0.5 * (l_swd_ll + l_swd_hf)
        total = total + sub * (mask.sum().float() / B)

    if b1_encoder is not None and b1_protos is not None:
        emb, _ = b1_encoder(z_pred.float())
        l_enc = (1.0 - (emb * b1_protos[tgt_ids]).sum(dim=-1)).mean()
        total = total + 0.35 * l_enc

    return total


def normalized_band_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    canon: ContentCanonicalizer,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5),
) -> torch.Tensor:
    """Subband-normalized MSE across (LL, LH, HL, HH) so high-energy LL does not dominate."""
    p_ll, p_lh, p_hl, p_hh = dwt2_haar(pred.float())
    t_ll, t_lh, t_hl, t_hh = dwt2_haar(target.float())

    l_ll = F.mse_loss(p_ll / canon.ll_std, t_ll / canon.ll_std)
    l_lh = F.mse_loss(p_lh / canon.lh_std, t_lh / canon.lh_std)
    l_hl = F.mse_loss(p_hl / canon.hl_std, t_hl / canon.hl_std)
    l_hh = F.mse_loss(p_hh / canon.hh_std, t_hh / canon.hh_std)

    w_ll, w_lh, w_hl, w_hh = weights
    return (w_ll * l_ll + w_lh * l_lh + w_hl * l_hl + w_hh * l_hh) / sum(weights)


@torch.inference_mode()
def generate_v6(
    model: C3UNet,
    canon: ContentCanonicalizer,
    z_src: torch.Tensor,
    tgt_ids: torch.Tensor,
    *,
    canon_mode: str = "c1",
    ode_steps: int = 8,
    sigma0: float = 0.0,
    seed: int = 42,
) -> torch.Tensor:
    """Inference for D1 (direct) or F1 (C³-Flow)."""
    c = canon(z_src, mode=canon_mode, augment=False)
    if model.mode == "direct":
        return model(c, 0.0, c, tgt_ids)

    if sigma0 > 0.0:
        g = torch.Generator(device=z_src.device).manual_seed(seed)
        eps = torch.randn(c.shape, generator=g, device=z_src.device, dtype=c.dtype)
        z = c + sigma0 * eps
    else:
        z = c.clone()

    dt = 1.0 / max(ode_steps, 1)
    for i in range(ode_steps):
        tau = i * dt
        v = model(z, tau, c, tgt_ids)
        z = z + dt * v
    return z


# ---------------------------------------------------------------------------
# Section IV: Mandatory 3-Part Conditioning Validity Audit
# ---------------------------------------------------------------------------

def audit_conditioning_validity(
    model: C3UNet,
    canon: ContentCanonicalizer,
    val_latents_by_style: Dict[str, torch.Tensor],
    canon_mode: str,
    style_targets: dict,
    coherence_beta: float,
    ode_steps: int = 8,
    sigma0: float = 0.15,
    device: str = "cuda",
) -> dict:
    """Run the 3 mandatory V6 Section IV checks:
    1. L_correct < L_wrong on held-out real paintings (with paired difference & 95% CI)
    2. 5-way Target Style Selectivity on identical canonical content c
    3. Content Condition Swap sensitivity under fixed Style-ID
    """
    model.eval()
    # 1. Training Condition Validity: L_correct vs L_wrong on 200 held-out real paintings (40/style)
    x_list, y_list = [], []
    for sid, s in enumerate(STYLES):
        x_list.append(val_latents_by_style[s][:40])
        y_list.append(torch.full((40,), sid, dtype=torch.long))
    z_val = torch.cat(x_list, 0).to(device)
    y_val = torch.cat(y_list, 0).to(device)
    z_tgt = make_coherent_real_target(z_val, y_val, style_targets, coherence_beta=coherence_beta)

    with torch.inference_mode():
        c_val = canon(z_val, mode=canon_mode, augment=False)
        torch.manual_seed(42)
        if model.mode == "direct":
            pred_corr = model(c_val, 0.0, c_val, y_val)
            target_tensor = z_tgt
        else:
            tau = torch.full((z_val.shape[0],), 0.5, device=device)
            z0 = c_val + sigma0 * torch.randn_like(c_val)
            z_tau = 0.5 * z0 + 0.5 * z_tgt
            pred_corr = model(z_tau, tau, c_val, y_val)
            target_tensor = z_tgt - z0

        # Per-sample normalized band error for correct ID
        def per_sample_err(pred, targ):
            p_ll, p_lh, p_hl, p_hh = dwt2_haar(pred.float())
            t_ll, t_lh, t_hl, t_hh = dwt2_haar(targ.float())
            e = (
                ((p_ll - t_ll) / canon.ll_std).pow(2).mean(dim=(1, 2, 3))
                + ((p_lh - t_lh) / canon.lh_std).pow(2).mean(dim=(1, 2, 3))
                + ((p_hl - t_hl) / canon.hl_std).pow(2).mean(dim=(1, 2, 3))
                + 0.5 * ((p_hh - t_hh) / canon.hh_std).pow(2).mean(dim=(1, 2, 3))
            ) / 3.5
            return e

        err_corr = per_sample_err(pred_corr, target_tensor)
        err_wrongs = []
        for shift in (1, 2, 3, 4):
            y_w = (y_val + shift) % 5
            if model.mode == "direct":
                pred_w = model(c_val, 0.0, c_val, y_w)
            else:
                pred_w = model(z_tau, tau, c_val, y_w)
            err_wrongs.append(per_sample_err(pred_w, target_tensor))
        err_wrong = torch.stack(err_wrongs, dim=0).mean(dim=0)
        diff = (err_wrong - err_corr).cpu().numpy()

    check1 = {
        "n_samples": int(z_val.shape[0]),
        "L_correct_mean": float(err_corr.mean().item()),
        "L_wrong_mean": float(err_wrong.mean().item()),
        "paired_diff_wrong_minus_correct": float(diff.mean()),
        "paired_diff_95ci": boot_ci(diff),
        "frac_L_correct_smaller": float((diff > 0).mean()),
        "passed": bool(diff.mean() > 0 and boot_ci(diff)[0] > 0),
    }

    # 2. Target Style Selectivity: same 25 canonical contents -> generate all 5 styles
    src_lat_dict = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted(src_lat_dict.keys())[:25]
    z_25 = torch.stack([src_lat_dict[k] for k in stems]).to(device)

    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = load_vae(device)
    trt = load_trt(vae, device)

    sim_matrix = np.zeros((5, 5), dtype=np.float64)
    all_gen_by_style = {}
    with torch.inference_mode():
        for tgt_sid in range(5):
            ids = torch.full((len(stems),), tgt_sid, device=device, dtype=torch.long)
            z_g = generate_v6(
                model, canon, z_25, ids,
                canon_mode=canon_mode, ode_steps=ode_steps, sigma0=sigma0, seed=42,
            )
            all_gen_by_style[tgt_sid] = z_g
            img_g = decode_trt(trt, vae, z_g, batch=8)
            u8 = img_g.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(u8[i]) for i in range(len(stems))]
            feats = csd_model.feats_pils(csd, pils, batch=16, device=device)  # (25, 768)
            sim_matrix[tgt_sid] = (feats @ proto.T).mean(axis=0)

    # Column-centered or raw diagonal vs off-diagonal selectivity
    diag_mean = float(np.diag(sim_matrix).mean())
    off_mean = float((sim_matrix.sum() - np.trace(sim_matrix)) / 20.0)
    # Also check how many of the 5 columns achieve their maximum on the diagonal
    col_argmax_matches = int(sum(int(sim_matrix[:, j].argmax() == j) for j in range(5)))
    # And pairwise output L2 distance across the 5 generated styles for the same content c
    pairwise_l2 = []
    for i in range(5):
        for j in range(i + 1, 5):
            d_ij = (all_gen_by_style[i] - all_gen_by_style[j]).pow(2).mean().sqrt().item()
            pairwise_l2.append(d_ij)

    check2 = {
        "n_contents": len(stems),
        "csd_5x5_matrix": sim_matrix.round(4).tolist(),
        "diag_mean": round(diag_mean, 4),
        "offdiag_mean": round(off_mean, 4),
        "selectivity_margin": round(diag_mean - off_mean, 4),
        "column_top1_matches_out_of_5": col_argmax_matches,
        "mean_cross_style_latent_l2": round(float(np.mean(pairwise_l2)), 4),
        "passed": bool((diag_mean - off_mean) > 0.015 and col_argmax_matches >= 3),
    }

    # 3. Content Condition Validity: swap content c_A and c_B under fixed Style-ID
    with torch.inference_mode():
        img_orig = decode_trt(trt, vae, z_25, batch=8)
        u8_o = img_orig.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_o = [Image.fromarray(u8_o[i]) for i in range(len(stems))]
        dcls_o, _ = encode_dino(dino, pils_o, batch=16, device=device)
        dcls_o = F.normalize(dcls_o.float(), dim=-1)

        # Take z_g generated for style 0 (or average over styles)
        z_g0 = all_gen_by_style[1]
        img_g0 = decode_trt(trt, vae, z_g0, batch=8)
        u8_g0 = img_g0.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_g0 = [Image.fromarray(u8_g0[i]) for i in range(len(stems))]
        dcls_g0, _ = encode_dino(dino, pils_g0, batch=16, device=device)
        dcls_g0 = F.normalize(dcls_g0.float(), dim=-1)

        sim_own = (dcls_g0 * dcls_o).sum(dim=-1).mean().item()
        dcls_swapped = torch.roll(dcls_o, shifts=1, dims=0)
        sim_swapped = (dcls_g0 * dcls_swapped).sum(dim=-1).mean().item()

        ssm_own = spatial_self_sim_corr(z_g0, z_25)
        ssm_swapped = spatial_self_sim_corr(z_g0, torch.roll(z_25, shifts=1, dims=0))

    check3 = {
        "dino_c_own_content": round(float(sim_own), 4),
        "dino_c_swapped_content": round(float(sim_swapped), 4),
        "dino_c_margin": round(float(sim_own - sim_swapped), 4),
        "selfsim_own_content": round(float(ssm_own), 4),
        "selfsim_swapped_content": round(float(ssm_swapped), 4),
        "passed": bool((sim_own - sim_swapped) > 0.25 and (ssm_own - ssm_swapped) > 0.15),
    }

    del csd, dino, vae, trt
    torch.cuda.empty_cache()

    return {
        "check1_training_condition_validity": check1,
        "check2_target_style_selectivity": check2,
        "check3_content_condition_validity": check3,
        "all_three_passed": bool(check1["passed"] and check2["passed"] and check3["passed"]),
    }


# ---------------------------------------------------------------------------
# 200-Task Quick Eval & 750-Task Full Evaluation
# ---------------------------------------------------------------------------

def evaluate_200_v6(
    model: C3UNet,
    canon: ContentCanonicalizer,
    canon_mode: str = "c1",
    ode_steps: int = 8,
    sigma0: float = 0.0,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    model.eval()
    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    lat_dict = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted({t["src_stem"] for t in tasks})
    z_all = torch.stack([lat_dict[s] for s in stems]).to(device)
    stem_idx = {s: i for i, s in enumerate(stems)}
    src_idx = torch.tensor([stem_idx[t["src_stem"]] for t in tasks], device=device)
    tgt_ids = torch.tensor([t["tgt_style_id"] for t in tasks], device=device, dtype=torch.long)

    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    src_tensors = {}
    for t in tasks:
        if t["src_stem"] not in src_tensors:
            src_tensors[t["src_stem"]] = _load_eval_image_tensor(
                Path(t["src_path"]), size=512
            ).unsqueeze(0).to(device)

    csd_src50 = torch.load(V5_ROOT / "cache" / "eval_src_csd_feats.pt", map_location="cpu", weights_only=False)
    csd_src_cache = {s: np.asarray(csd_src50[s]) for s in stems}
    dino_src_cache = {}
    with torch.inference_mode():
        img_src = decode_trt(trt, vae, z_all, batch=8)
        u8_src = img_src.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_src = [Image.fromarray(u8_src[i]) for i in range(len(stems))]
        dcls, _ = encode_dino(dino, pils_src, batch=16, device=device)
        for i, s in enumerate(stems):
            dino_src_cache[s] = dcls[i].float().cpu().numpy()

        # Also evaluate 50 same-style reconstructions!
        same_ids = torch.tensor(
            [next(t["src_style_id"] for t in tasks if t["src_stem"] == s) for s in stems],
            device=device, dtype=torch.long,
        )
        z_same = generate_v6(
            model, canon, z_all, same_ids,
            canon_mode=canon_mode, ode_steps=ode_steps, sigma0=sigma0, seed=seed,
        )
        img_same = decode_trt(trt, vae, z_same, batch=8)
        u8_same = img_same.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_same = [Image.fromarray(u8_same[i]) for i in range(len(stems))]
        f_same = csd_model.feats_pils(csd, pils_same, batch=16, device=device)
        d_same, _ = encode_dino(dino, pils_same, batch=16, device=device)
        d_same = d_same.float().cpu().numpy()
        same_dS = [
            float(f_same[i] @ proto[int(same_ids[i].item())] - csd_src_cache[stems[i]] @ proto[int(same_ids[i].item())])
            for i in range(len(stems))
        ]
        same_dino_c = [float(d_same[i] @ dino_src_cache[stems[i]]) for i in range(len(stems))]

        # Cross-style 200 tasks
        z_src_200 = z_all[src_idx]
        z_gen = generate_v6(
            model, canon, z_src_200, tgt_ids,
            canon_mode=canon_mode, ode_steps=ode_steps, sigma0=sigma0, seed=seed,
        )
        img = decode_trt(trt, vae, z_gen, batch=8)
        u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        gen01 = u8.float().div_(255.0)
        arr_gen = u8.permute(0, 2, 3, 1).numpy()
        pils = [Image.fromarray(arr_gen[i]) for i in range(len(tasks))]

        fg = csd_model.feats_pils(csd, pils, batch=16, device=device)
        dg, _ = encode_dino(dino, pils, batch=16, device=device)
        dg = dg.float().cpu().numpy()
        lps = []
        for s in range(0, len(tasks), 16):
            chunk = gen01[s : s + 16].to(device)
            src = torch.cat([src_tensors[t["src_stem"]] for t in tasks[s : s + 16]], 0)
            lps.append(lpips_fn(to_lpips_input(chunk), to_lpips_input(src)).flatten().cpu())
        lps = torch.cat(lps, 0).numpy()

    rows = []
    for i, t in enumerate(tasks):
        fs = csd_src_cache[t["src_stem"]]
        tgt, sid = t["tgt_style_id"], t["src_style_id"]
        scores = fg[i] @ proto.T                      # (5,) cosine to each prototype
        dT = float(scores[tgt] - fs @ proto[tgt])
        dS = float(scores[sid] - fs @ proto[sid])
        other = np.delete(scores, tgt)
        rows.append({
            "task_id": t["task_id"], "src_style": t["src_style"], "tgt_style": t["tgt_style"],
            "dT": dT, "dS": dS, "dM": dT - dS,
            "target_margin": float(scores[tgt] - other.max()),
            "target_top1": int(scores.argmax() == tgt),
            "pred_class": int(scores.argmax()),
            "dino_c": float(dg[i] @ dino_src_cache[t["src_stem"]]),
            "lpips": float(lps[i]),
        })

    dT_arr = np.array([r["dT"] for r in rows])
    dS_arr = np.array([r["dS"] for r in rows])
    per_style = {
        s: float(np.mean([r["dT"] for r in rows if r["tgt_style"] == s]))
        for s in STYLES
    }
    # 5x5 confusion matrix: rows = target style, cols = predicted (CSD prototype argmax).
    # The diagonal is the target Top-1 per style; off-diagonal mass shows which style a
    # failing target collapses into, which a single macro average hides.
    conf = np.zeros((len(STYLES), len(STYLES)), dtype=int)
    for r in rows:
        conf[STYLES.index(r["tgt_style"]), r["pred_class"]] += 1
    per_style_top1 = {
        s: float(np.mean([r["target_top1"] for r in rows if r["tgt_style"] == s]))
        for s in STYLES
    }
    per_style_margin = {
        s: float(np.mean([r["target_margin"] for r in rows if r["tgt_style"] == s]))
        for s in STYLES
    }
    del csd, dino, vae, trt, lpips_fn
    torch.cuda.empty_cache()

    return {
        "n_cross": len(rows),
        "mean_dT": float(dT_arr.mean()),
        "dT_ci": boot_ci(dT_arr),
        "mean_dS": float(dS_arr.mean()),
        "mean_dM": float((dT_arr - dS_arr).mean()),
        "mean_dino_c": float(np.mean([r["dino_c"] for r in rows])),
        "mean_lpips": float(np.mean([r["lpips"] for r in rows])),
        "frac_dT_positive": float((dT_arr > 0).mean()),
        "per_target_dT": per_style,
        "per_target_top1": per_style_top1,
        "per_target_margin": per_style_margin,
        "confusion_target_vs_pred": conf.tolist(),
        "confusion_styles": STYLES,
        "styles_improved": int(sum(1 for v in per_style.values() if v > 0)),
        "ukiyo_dT": per_style["Ukiyo_e"],
        "same50_mean_dS": float(np.mean(same_dS)),
        "same50_mean_dino_c": float(np.mean(same_dino_c)),
    }


def render_and_eval_750_v6(
    model: C3UNet,
    canon: ContentCanonicalizer,
    cell: str,
    *,
    canon_mode: str = "c1",
    ode_steps: int = 8,
    sigma0: float = 0.0,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Full 600 cross + 150 same evaluation under the exact frozen V4/V5 protocol."""
    out_dir = RENDER_ROOT / cell
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    from tools.v5.route_c import source_latents
    z_src, src_paths, requests, style_meta = source_latents(device, max_src=30)
    num_styles = len(style_meta)
    req_map = {(str(r["src_path"]), int(r["tgt_style_id"])): r for r in requests}

    vae = load_vae(device)
    trt = load_trt(vae, device)
    ns = argparse.Namespace(
        disable_clip=False, disable_lpips=False, disable_dino=False,
        test_dir=str(REPO_ROOT / "data" / "test"),
        clip_model="openai/clip-vit-base-patch32", clip_local_only=True,
        clip_cache_dir=str(paths.eval_cache_root("hf") / "hub"), clip_skip_processor=False,
        amp=False, lpips_net="alex", lpips_chunk=32,
        dino_model="facebook/dinov2-small",
        dino_cache_dir=str(paths.eval_cache_root("hf")), allow_network=False,
        ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=512,
        metric_batch=32, ref_batch=32, only_clip_style=True,
        clip_style_idt_baseline=0.0, clip_vectors=False,
    )
    engine = MetricEngine(ns, {}, style_meta, device)
    engine.load(Timer())
    engine.prepare(Timer(), src_paths)

    rows_all = []
    save_pool = ThreadPoolExecutor(max_workers=8)
    futs = []
    t_start = time.perf_counter()

    for b in range(0, len(src_paths), 8):
        chunk_paths = src_paths[b : b + 8]
        z_s = z_src[b : b + len(chunk_paths)]
        n = z_s.shape[0]
        for tgt in range(num_styles):
            rows = [req_map[(str(p), tgt)] for p in chunk_paths]
            ids = torch.full((n,), tgt, dtype=torch.long, device=device)
            z_gen = generate_v6(
                model, canon, z_s, ids,
                canon_mode=canon_mode, ode_steps=ode_steps, sigma0=sigma0, seed=seed + b + tgt,
            )
            img = decode_trt(trt, vae, z_gen, batch=8)
            u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8)
            gen01 = u8.float().div_(255.0)
            arr = u8.cpu().numpy()
            for i, r in enumerate(rows):
                futs.append(
                    save_pool.submit(
                        lambda a, p: Image.fromarray(a).save(p, format="PNG"),
                        arr[i].transpose(1, 2, 0).copy(),
                        img_dir / r["gen_name"],
                    )
                )
            out_rows, _ = engine.batch_metrics(gen01, rows, Timer())
            rows_all.extend(out_rows)
        if (b // 8) % 5 == 0:
            print(f"[eval750:{cell}] {min(b+8, len(src_paths))}/{len(src_paths)} sources ({time.perf_counter()-t_start:.0f}s)", flush=True)

    for f in futs:
        f.result()
    save_pool.shutdown(wait=True)

    write_csv(out_dir / "metrics.csv", rows_all, METRIC_HEADER)
    write_csv(out_dir / "dino_metrics.csv", rows_all, DINO_HEADER)
    del engine, vae, trt
    torch.cuda.empty_cache()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "pseudo_gt" / "v4_eval750.py"),
        "--cell", cell,
        "--render-root", str(RENDER_ROOT),
        "--max-src", "30",
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    eval750 = json.loads((out_dir / "v4_eval750.json").read_text(encoding="utf-8"))
    rows750 = json.loads((out_dir / "v4_eval750_rows.json").read_text(encoding="utf-8"))
    cross_rows = [r for r in rows750 if r["cross"]]
    per_tgt = {
        s: float(np.mean([r["csd_dT"] for r in cross_rows if r["tgt_style"] == s]))
        for s in STYLES
    }
    eval750["per_target_cross_CSD_dT"] = per_tgt
    eval750["styles_improved"] = int(sum(1 for v in per_tgt.values() if v > 0))
    eval750["ukiyo_dT"] = per_tgt["Ukiyo_e"]
    (out_dir / "v4_eval750.json").write_text(json.dumps(eval750, indent=2), encoding="utf-8")
    return eval750


# ---------------------------------------------------------------------------
# Main Training Loop for D1 / F1
# ---------------------------------------------------------------------------

def train_and_evaluate(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V6_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = V6_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_root = paths.asset_root()
    orig_dir = asset_root / "pseudo_gt_24h" / "orig_mean_fp16"

    latents_by_style = {}
    train_by_style = {}
    val_by_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(orig_dir / f"{sid:02d}_{s}.pt", map_location="cpu", weights_only=False)
        x = d["latents"].float()
        latents_by_style[s] = x
        val_by_style[s] = x[:100]
        train_by_style[s] = x[100:]

    global_stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(global_stats).to(device)
    # Plan L4: Core64 vs a larger reference pool. "Full" is infeasible in memory
    # (8,989 Impressionism latents alone is ~590 MB on GPU), so 1024/style is used as
    # the largest practical pool and the cap is recorded.
    ref_top_k = args.ref_top_k if args.ref_top_k > 0 else (1024 if args.ref_full else 64)
    style_targets = build_indomain_style_targets(train_by_style, asset_root, device=device,
                                                 top_k=ref_top_k)
    print(f"[{args.name}] reference pool: top_k={ref_top_k}/style "
          f"(Core64={ref_top_k == 64})", flush=True)

    # Load frozen in-domain B1 LatentStyleEncoder (trained only on D5 real latents)
    b1_ckpt_path = V5_ROOT / "b1_style_encoder" / "latent_style_encoder.pt"
    b1_encoder, b1_protos = None, None
    b1_status = "disabled_by_flag" if args.no_b1 else (
        "missing_ckpt" if not b1_ckpt_path.exists() else
        "disabled_by_w_style_dist" if args.w_style_dist <= 0 else "loaded")
    if b1_status == "loaded":
        from tools.v5.b1_latent_style_encoder import LatentStyleEncoder
        b1_ck = torch.load(b1_ckpt_path, map_location="cpu", weights_only=False)
        b1_encoder = LatentStyleEncoder().to(device).eval()
        b1_encoder.load_state_dict(b1_ck["model"])
        for p in b1_encoder.parameters():
            p.requires_grad_(False)
        b1_protos = torch.stack([b1_ck["prototypes"][s] for s in STYLES]).to(device)
    # P0 requirement: prove whether the B1 branch is actually active, instead of
    # assuming it. A missing checkpoint used to be silent.
    print(f"[{args.name}] B1 branch: {b1_status} "
          f"(ckpt_exists={b1_ckpt_path.exists()}, w_style_dist={args.w_style_dist})",
          flush=True)

    # Move training tensors per style to GPU (~900 MiB total)
    gpu_train_all = {s: train_by_style[s].to(device) for s in STYLES}
    gpu_train_core = {s: style_targets["styles"][s]["core_latents"].float().to(device) for s in STYLES}

    torch.manual_seed(args.seed)
    model = C3UNet(base_dim=args.base_dim, num_styles=len(STYLES), mode=args.mode,
                   no_content=args.no_content, no_style=args.no_style,
                   content_mode=args.content_mode,
                   inject_64=not args.no_inject_64,
                   inject_32=not args.no_inject_32,
                   inject_16=not args.no_inject_16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Plan section 5: decoupled absolute coefficients.
    lam_s = args.lambda_cross if args.lambda_cross >= 0 else args.w_cross
    lam_c = (args.lambda_canon if args.lambda_canon >= 0
             else args.w_cross * args.w_canon_inv)
    lam_same = 0.0 if args.no_same_style else args.w_style_dist
    print(f"[{args.name}] mode={args.mode} params={n_params/1e6:.2f}M canon={args.canon_mode} "
          f"content_mode={args.content_mode} no_style={args.no_style} "
          f"inject=({model.inject_64},{model.inject_32},{model.inject_16}) "
          f"lam_same={lam_same} lam_s={lam_s} lam_c={lam_c} "
          f"steps={args.steps} beta={args.coherence_beta}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)

    per_style_mb = max(4, args.microbatch // len(STYLES))
    accum = args.accum
    loss_hist = []
    diag = {}
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss_val = 0.0

        for _ in range(accum):
            # Class-balanced sampling: 50% from full style pool, 50% from core style pool
            z_chunks, y_chunks = [], []
            for sid, s in enumerate(STYLES):
                if args.core_sampler_off:
                    # plan L3: 0% from the Core64 pool; reference stats still come from
                    # Core64, so this isolates the sampling question from the target one.
                    n_core = 0
                    n_all = per_style_mb
                else:
                    n_core = per_style_mb // 2
                    n_all = per_style_mb - n_core
                pool_c = gpu_train_core[s]
                pool_a = gpu_train_all[s]
                parts = []
                if n_core > 0:
                    idx_c = torch.randint(0, pool_c.shape[0], (n_core,), device=device)
                    parts.append(pool_c[idx_c])
                if n_all > 0:
                    idx_a = torch.randint(0, pool_a.shape[0], (n_all,), device=device)
                    parts.append(pool_a[idx_a])
                z_chunks.append(torch.cat(parts, dim=0))
                y_chunks.append(torch.full((per_style_mb,), sid, device=device, dtype=torch.long))

            z_real = torch.cat(z_chunks, dim=0)
            y_real = torch.cat(y_chunks, dim=0)

            with torch.no_grad():
                c_in = canon(z_real, mode=args.canon_mode, augment=True)
                z_target = make_coherent_real_target(
                    z_real, y_real, style_targets, coherence_beta=args.coherence_beta
                )

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if args.mode == "direct":
                    z_pred = model(c_in, 0.0, c_in, y_real)
                    l_rec = normalized_band_mse(z_pred, z_target, canon)
                    z1_est = z_pred
                else:
                    eps = torch.randn_like(c_in)
                    z0 = c_in + args.sigma0 * eps
                    tau = torch.rand(z_real.shape[0], device=device)
                    tv = tau.view(-1, 1, 1, 1)
                    z_tau = (1.0 - tv) * z0 + tv * z_target
                    v_star = z_target - z0
                    v_pred = model(z_tau, tau, c_in, y_real)
                    l_rec = normalized_band_mse(v_pred, v_star, canon)
                    z1_est = z0 + v_pred

                l_style_same = compute_style_distribution_loss(
                    z1_est, y_real, style_targets, b1_encoder=b1_encoder, b1_protos=b1_protos
                )
                loss = args.w_rec * l_rec + lam_same * l_style_same

                # P0 requirement: prove the B1/style-distribution branch actually carries
                # gradient into the generator on a real batch, instead of inferring it
                # from a non-zero loss value.
                if step == 1 and "b1_grad" not in diag:
                    _gp = [p for p in model.parameters() if p.requires_grad]

                    def _gn(t):
                        gs = torch.autograd.grad(t, _gp, retain_graph=True,
                                                 allow_unused=True)
                        s = torch.zeros((), device=device)
                        for g in gs:
                            if g is not None:
                                s = s + g.detach().float().square().sum()
                        return float(s.sqrt())

                    diag["b1_loss"] = float(l_style_same)
                    diag["b1_grad"] = _gn(lam_same * l_style_same)
                    diag["rec_grad"] = _gn(args.w_rec * l_rec)
                    diag["b1_status"] = b1_status
                    print(f"[{args.name}] step1 diag: L_style_same={diag['b1_loss']:.6f} "
                          f"grad(lam_same*L_same)={diag['b1_grad']:.6e} "
                          f"grad(L_rec)={diag['rec_grad']:.6e} "
                          f"ratio={diag['b1_grad']/max(diag['rec_grad'],1e-12):.4f} "
                          f"b1={b1_status}", flush=True)

                if lam_s > 0.0 or lam_c > 0.0:
                    # Cross-style pass on the exact same canonical content c_in with shifted target Style-ID
                    shift = torch.randint(1, len(STYLES), (z_real.shape[0],), device=device)
                    y_cross = (y_real + shift) % len(STYLES)
                    if args.mode == "direct":
                        z_cross = model(c_in, 0.0, c_in, y_cross)
                    else:
                        v_cross = model(z0, torch.zeros_like(tau), c_in, y_cross)
                        z_cross = z0 + v_cross

                    l_style_cross = compute_style_distribution_loss(
                        z_cross, y_cross, style_targets, b1_encoder=b1_encoder, b1_protos=b1_protos
                    )
                    # Content-canonical invariance: canonicalizing z_cross must recover c(z_real)
                    c_clean = canon(z_real, mode="c1", augment=False)
                    c_cross = canon(z_cross.float(), mode="c1", augment=False)
                    c_ll_real, c_lh_real, c_hl_real, _ = dwt2_haar(c_clean)
                    c_ll_gen, c_lh_gen, c_hl_gen, _ = dwt2_haar(c_cross)
                    l_canon_inv = (
                        F.mse_loss(c_ll_gen, c_ll_real)
                        + 0.35 * (F.mse_loss(c_lh_gen, c_lh_real) + F.mse_loss(c_hl_gen, c_hl_real))
                    )
                    loss = loss + lam_s * l_style_cross + lam_c * l_canon_inv

                loss_scaled = loss / accum

            loss_scaled.backward()
            step_loss_val += float(loss_scaled.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_hist.append(step_loss_val)

        if step == 1 or step % 200 == 0 or step == args.steps:
            peak_mib = torch.cuda.max_memory_allocated(0) / (1024**2)
            print(f"[{args.name}] step {step}/{args.steps} loss={np.mean(loss_hist[-100:]):.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} peak_vram={peak_mib:.0f}MiB "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    train_wall = round(time.time() - t0, 1)
    peak_vram = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1)

    ckpt_path = out_dir / f"{args.name}.pt"
    torch.save({
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": vars(args),
        "n_params": int(n_params),
        "train_wall_sec": train_wall,
        "peak_vram_mib": peak_vram,
        "final_loss": float(np.mean(loss_hist[-50:])),
    }, ckpt_path)

    print(f"[{args.name}] Running Section IV 3-part conditioning validity audit...", flush=True)
    cond_audit = audit_conditioning_validity(
        model, canon, val_by_style, args.canon_mode, style_targets,
        coherence_beta=args.coherence_beta, ode_steps=args.ode_steps, sigma0=args.sigma0, device=device,
    )
    print(f"[{args.name}] Conditioning audit: all_three_passed={cond_audit['all_three_passed']} "
          f"L_diff={cond_audit['check1_training_condition_validity']['paired_diff_wrong_minus_correct']:+.4f} "
          f"sel_margin={cond_audit['check2_target_style_selectivity']['selectivity_margin']:+.4f} "
          f"content_margin={cond_audit['check3_content_condition_validity']['dino_c_margin']:+.4f}", flush=True)

    print(f"[{args.name}] Running 200-task cross-style + 50 same-style evaluation...", flush=True)
    eval200 = evaluate_200_v6(
        model, canon, canon_mode=args.canon_mode, ode_steps=args.ode_steps,
        sigma0=args.sigma0, seed=args.seed, device=device,
    )
    print(f"[{args.name}] Eval200: CSD_dT={eval200['mean_dT']:+.4f} dM={eval200['mean_dM']:+.4f} "
          f"DINO-C={eval200['mean_dino_c']:.4f} LPIPS={eval200['mean_lpips']:.4f} "
          f"styles+={eval200['styles_improved']}/5 ukiyo={eval200['ukiyo_dT']:+.4f} "
          f"same50_dS={eval200['same50_mean_dS']:+.4f}", flush=True)

    report = {
        "name": args.name,
        "mode": args.mode,
        "canon_mode": args.canon_mode,
        "n_params": int(n_params),
        "steps": args.steps,
        "effective_batch": per_style_mb * len(STYLES) * accum,
        "coherence_beta": args.coherence_beta,
        "no_content": bool(args.no_content),
        "content_mode": args.content_mode,
        "no_style": bool(args.no_style),
        "w_style_dist": args.w_style_dist,
        "w_cross": args.w_cross,
        "w_canon_inv": args.w_canon_inv,
        "lambda_cross_abs": lam_s,
        "lambda_canon_abs": lam_c,
        "lambda_same_abs": lam_same,
        "b1_status": b1_status,
        "ref_top_k": ref_top_k,
        "core_sampler_off": bool(args.core_sampler_off),
        "inject_64": model.inject_64,
        "inject_32": model.inject_32,
        "inject_16": model.inject_16,
        "grad_diag": diag,
        "ode_steps": args.ode_steps,
        "sigma0": args.sigma0,
        "seed": args.seed,
        "train_wall_sec": train_wall,
        "peak_vram_mib": peak_vram,
        "final_loss": float(np.mean(loss_hist[-50:])),
        "conditioning_validity": cond_audit,
        "eval200": eval200,
    }
    if args.run_full750:
        print(f"[{args.name}] Running full 750 (600 cross + 150 same) evaluation...", flush=True)
        eval750 = render_and_eval_750_v6(
            model, canon, f"{args.name}_s{args.seed}",
            canon_mode=args.canon_mode, ode_steps=args.ode_steps, sigma0=args.sigma0,
            seed=args.seed, device=device,
        )
        report["eval750"] = eval750

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["direct", "flow"], required=True)
    ap.add_argument("--canon-mode", choices=["c1", "c1_gentle", "c1_soft", "c2", "none"],
                    default="c1")
    ap.add_argument("--no-content", action="store_true",
                    help="ablation: zero the per-block content injection (content is "
                         "still visible to the stem). Same weights/params.")
    ap.add_argument("--content-mode", choices=["full", "no_inject", "none"],
                    default="full",
                    help="ablation: 'none' also zeros z_in, i.e. a true "
                         "style-only-generation control")
    ap.add_argument("--no-style", action="store_true",
                    help="ablation: pin Style-ID to 0 (same weights/params)")
    ap.add_argument("--base-dim", type=int, default=56)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--microbatch", type=int, default=80)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--coherence-beta", type=float, default=0.65)
    ap.add_argument("--w-rec", type=float, default=1.0)
    ap.add_argument("--w-style-dist", type=float, default=0.5)
    ap.add_argument("--w-cross", type=float, default=0.5)
    ap.add_argument("--w-canon-inv", type=float, default=1.5)
    # --- plan section 5: decouple the product form lam_c = w_cross * w_canon_inv ---
    # Pass absolute coefficients to get L = L_rec + 0.25 L_same + lam_s L_cross + lam_c L_inv.
    # -1 means "not set" -> fall back to the historical product form.
    ap.add_argument("--lambda-cross", type=float, default=-1.0,
                    help="absolute lam_s. Overrides --w-cross when >= 0")
    ap.add_argument("--lambda-canon", type=float, default=-1.0,
                    help="absolute lam_c. Overrides --w-cross*--w-canon-inv when >= 0")
    ap.add_argument("--no-same-style", action="store_true",
                    help="plan L5: set lam_same=0, keep L_rec")
    ap.add_argument("--no-b1", action="store_true",
                    help="plan N6: drop the B1 latent-style-encoder branch only")
    ap.add_argument("--core-sampler-off", action="store_true",
                    help="plan L3: sample 0%% from the Core64 pool (reference stats unchanged)")
    ap.add_argument("--ref-full", action="store_true",
                    help="plan L4: build style reference stats from a much larger train "
                         "pool (1024/style) instead of Core64 (sampling unchanged)")
    ap.add_argument("--ref-top-k", type=int, default=0,
                    help="explicit reference-pool size per style; overrides --ref-full")
    ap.add_argument("--no-inject-64", action="store_true", help="plan S1")
    ap.add_argument("--no-inject-32", action="store_true", help="plan S2")
    ap.add_argument("--no-inject-16", action="store_true", help="plan S3")
    ap.add_argument("--ode-steps", type=int, default=8)
    ap.add_argument("--sigma0", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-full750", action="store_true")
    args = ap.parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
