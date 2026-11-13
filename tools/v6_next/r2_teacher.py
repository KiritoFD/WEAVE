"""WEAVE R2 Teacher and Gradient Analysis Toolkit.

Implements Sections P1 & P2 of WEAVE R2 12-Hour Execution Plan:
- R1 target margin gradient: g = ∇_z m_t(z_0) in fp32.
- Per-sample RMS normalization: g / (RMS(g) + 1e-12).
- Haar wavelet subband filtering (LL-only, HF-only, all) with energy conservation.
- Latent spatial self-similarity loss L_ss(z, zs) on 16x16 pooled features.
- Canonical content invariance loss L_inv(z, zs).
- Content tangential gradient projection: d = g_s - max(0, <g_s, g_c>/||g_c||^2) * g_c.
- Multi-vector directional alignment diagnostics (cosines with Q2a delta, -∇L_ss, -∇L_inv, ScoreDiff).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar


def compute_r1_gradient(
    z: torch.Tensor,
    target_style_id: torch.Tensor,
    critic: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute R1 target margin gradient g = ∇_z m_t(z) in fp32.

    Returns:
        g: (B, 4, 64, 64) gradient tensor
        m_t: (B,) margin values before gradient step
    """
    with torch.enable_grad():
        zr = z.detach().clone().float().requires_grad_(True)
        q = critic(zr).float()
        oh = F.one_hot(target_style_id.long(), q.shape[1]).bool()
        # m_t = q_t - logsumexp_{k != t} q_k
        other_max = torch.logsumexp(q.masked_fill(oh, -1e9), dim=1)
        m = q.gather(1, target_style_id.long()[:, None]).squeeze(1) - other_max
        g = torch.autograd.grad(m.sum(), zr)[0]
    return g.detach(), m.detach()


def rms_normalize(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample latent RMS normalization: g / (RMS(g) + eps).

    RMS(g) = sqrt( mean(g^2) ) = ||g|| / sqrt(D).
    """
    rms = g.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().view(-1, 1, 1, 1)
    return g / (rms + eps)


def filter_haar_subbands(g: torch.Tensor, band: str = "all") -> torch.Tensor:
    """Filter gradient into subbands using exact Haar DWT/IDWT.

    Args:
        g: (B, 4, 64, 64) gradient tensor
        band: 'all', 'll', or 'hf'
    Returns:
        Filtered gradient renormalized to the same per-sample RMS as g.
    """
    if band == "all":
        return g

    orig_rms = (
        g.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().view(-1, 1, 1, 1)
    )
    ll, lh, hl, hh = dwt2_haar(g.float())

    if band == "ll":
        zero = torch.zeros_like(lh)
        filtered = idwt2_haar(ll, zero, zero, zero)
    elif band == "hf":
        zero = torch.zeros_like(ll)
        filtered = idwt2_haar(zero, lh, hl, hh)
    else:
        raise ValueError(f"Unknown band: {band}")

    # Renormalize to original RMS to ensure identical editing magnitude
    filt_rms = (
        filtered.flatten(1)
        .pow(2)
        .mean(dim=1, keepdim=True)
        .sqrt()
        .view(-1, 1, 1, 1)
    )
    return (filtered / (filt_rms + 1e-12)) * orig_rms


def subband_energies(delta: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute energy and proportion in each of the 4 Haar subbands."""
    ll, lh, hl, hh = dwt2_haar(delta.float())
    e_ll = ll.pow(2).sum(dim=(1, 2, 3))
    e_lh = lh.pow(2).sum(dim=(1, 2, 3))
    e_hl = hl.pow(2).sum(dim=(1, 2, 3))
    e_hh = hh.pow(2).sum(dim=(1, 2, 3))
    total = (e_ll + e_lh + e_hl + e_hh).clamp(min=1e-12)
    return {
        "energy_ll": e_ll,
        "energy_lh": e_lh,
        "energy_hl": e_hl,
        "energy_hh": e_hh,
        "prop_ll": e_ll / total,
        "prop_lh": e_lh / total,
        "prop_hl": e_hl / total,
        "prop_hh": e_hh / total,
    }


def spatial_self_similarity_matrix(z: torch.Tensor) -> torch.Tensor:
    """Compute 16x16 spatial self-similarity matrix S(z).

    z: (B, 4, 64, 64) -> adaptive pool to (B, 4, 16, 16) -> (B, 256, 4)
    Normalized across channels, S(z) = P @ P^T: (B, 256, 256)
    """
    p = F.adaptive_avg_pool2d(z.float(), (16, 16))
    p = p.flatten(2).permute(0, 2, 1)  # (B, 256, 4)
    p = F.normalize(p, dim=-1)
    return torch.bmm(p, p.transpose(1, 2))  # (B, 256, 256)


def compute_content_losses(
    z: torch.Tensor,
    zs: torch.Tensor,
    canon_fn: Optional[torch.nn.Module] = None,
    canon_mode: str = "c1_gentle",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute self-similarity loss L_ss and canonical invariance loss L_inv.

    Returns:
        L_ss: (B,) MSE between spatial self-similarity matrices
        L_inv: (B,) MSE between canonicalized latents
    """
    s_z = spatial_self_similarity_matrix(z)
    s_zs = spatial_self_similarity_matrix(zs)
    l_ss = (s_z - s_zs).pow(2).mean(dim=(1, 2))

    if canon_fn is not None:
        c_z = canon_fn(z.float(), mode=canon_mode, augment=False)
        c_zs = canon_fn(zs.float(), mode=canon_mode, augment=False)
        l_inv = (c_z - c_zs).pow(2).mean(dim=(1, 2, 3))
    else:
        l_inv = torch.zeros_like(l_ss)

    return l_ss, l_inv


def compute_content_gradient(
    z0: torch.Tensor,
    zs: torch.Tensor,
    canon_fn: Optional[torch.nn.Module] = None,
    canon_mode: str = "c1_gentle",
    a: float = 1.0,
    b: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute content structure gradient g_c = ∇_z L_C(z0, zs).

    L_C = a * L_ss + b * L_inv.
    Returns:
        g_c: (B, 4, 64, 64) gradient tensor
        L_C: (B,) content structure loss at z0
    """
    with torch.enable_grad():
        zr = z0.detach().clone().float().requires_grad_(True)
        l_ss, l_inv = compute_content_losses(zr, zs, canon_fn, canon_mode)
        # Normalize by typical scales (L_ss ~ 0.05, L_inv ~ 0.10) if a, b = 1.0
        l_c = a * (l_ss / 0.05) + b * (l_inv / 0.10)
        g_c = torch.autograd.grad(l_c.sum(), zr)[0]
    return g_c.detach(), l_c.detach()


def project_tangential(
    g_s: torch.Tensor, g_c: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Half-space projection: remove component of g_s that increases content loss L_C.

    d = g_s - max(0, <g_s, g_c> / (||g_c||^2 + eps)) * g_c.
    Per-sample projection.
    """
    B = g_s.shape[0]
    g_s_flat = g_s.flatten(1)
    g_c_flat = g_c.flatten(1)

    dot = (g_s_flat * g_c_flat).sum(dim=1, keepdim=True)  # (B, 1)
    g_c_norm_sq = g_c_flat.pow(2).sum(dim=1, keepdim=True)  # (B, 1)

    coeff = torch.clamp(dot / (g_c_norm_sq + eps), min=0.0)  # (B, 1)
    d_flat = g_s_flat - coeff * g_c_flat
    return d_flat.view_as(g_s)


def cosine_similarity(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine similarity between two tensors: (B,)."""
    u_f = u.detach().flatten(1).float()
    v_f = v.detach().flatten(1).float()
    dot = (u_f * v_f).sum(dim=1)
    norm_u = u_f.norm(dim=1).clamp(min=1e-12)
    norm_v = v_f.norm(dim=1).clamp(min=1e-12)
    return dot / (norm_u * norm_v)
