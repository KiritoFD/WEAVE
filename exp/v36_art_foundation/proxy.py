"""Zero-DINO Multi-Transform Destylized Proxy Generator for V36 T2/T3 Pairs (Spec §3.2).

Applies 5 fast GPU pixel-space style-stripping perturbations to break source style coupling:
  1. Random Oklab palette / chroma moment redistribution
  2. Low-frequency luminance moment redistribution
  3. Local contrast jitter
  4. Edge-preserving detail smoothing (removes brushstrokes while preserving contours)
  5. High-frequency band-limited texture jitter
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.structure import rgb_to_oklab


def _oklab_to_srgb(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    l_ = L + 0.3963377774 * ab[:, 0:1] + 0.2158037573 * ab[:, 1:2]
    m_ = L - 0.1055613458 * ab[:, 0:1] - 0.0638541728 * ab[:, 1:2]
    s_ = L - 0.0894841775 * ab[:, 0:1] - 1.2914855480 * ab[:, 1:2]
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
    lin = torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)
    srgb = torch.where(
        lin <= 0.0031308,
        12.92 * lin,
        1.055 * torch.pow(lin.clamp_min(1e-6), 1.0 / 2.4) - 0.055,
    )
    return srgb.clamp(0.0, 1.0)


@torch.no_grad()
def destylize_proxy_v36(
    rgb01: torch.Tensor,
    *,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Create one stochastic structure-preserving, style-perturbed proxy in [0, 1]."""
    x = rgb01.float().clamp(0.0, 1.0)
    B, _, H, W = x.shape
    dev = x.device
    lab = rgb_to_oklab(x)
    L = lab[:, 0:1]
    ab = lab[:, 1:3]

    # 1. Random Oklab palette / chroma moment matching (strip original palette while keeping non-degenerate color)
    ab_mu = ab.mean(dim=(2, 3), keepdim=True)
    ab_sd = ab.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-3)
    ab_norm = ((ab - ab_mu) / ab_sd).clamp(-4.0, 4.0)
    tgt_ab_mu = torch.empty(B, 2, 1, 1, device=dev).uniform_(-0.04, 0.04, generator=gen)
    tgt_ab_sd = torch.empty(B, 2, 1, 1, device=dev).uniform_(0.008, 0.045, generator=gen)
    ab_out = ab_norm * tgt_ab_sd + tgt_ab_mu

    # 2. Edge-preserving multi-scale smoothing (attenuate strokes, keep major boundaries)
    L_s = F.interpolate(F.avg_pool2d(L, 4, 4), size=(H, W), mode="bilinear", align_corners=False)
    L_mid = F.interpolate(F.avg_pool2d(L, 2, 2), size=(H, W), mode="bilinear", align_corners=False)
    detail = L - L_mid
    edge_gate = (detail.abs() * 8.0).clamp(0.0, 1.0)
    L_ep = 0.60 * L_mid + 0.40 * L_s + 0.25 * edge_gate * detail

    # 3. Low-frequency luminance moment redistribution + local contrast jitter
    L_bg = F.interpolate(F.avg_pool2d(L_ep, 16, 16), size=(H, W), mode="bilinear", align_corners=False)
    L_local = L_ep - L_bg
    mu_t = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.42, 0.58, generator=gen)
    bg_scale = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.45, 0.90, generator=gen)
    lc_scale = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.72, 1.22, generator=gen)
    L_bg_new = mu_t + bg_scale * (L_bg - L_bg.mean(dim=(2, 3), keepdim=True))
    L_out = L_bg_new + lc_scale * L_local

    # 4. Mild high-frequency texture noise perturbation to disrupt residual stroke regularity
    raw_noise = torch.randn(B, 1, H, W, device=dev, generator=gen)
    hf_noise = raw_noise - F.interpolate(F.avg_pool2d(raw_noise, 4, 4), size=(H, W), mode="bilinear", align_corners=False)
    hf_amp = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.015, 0.040, generator=gen)
    L_out = (L_out + hf_amp * hf_noise).clamp(0.02, 0.98)

    return _oklab_to_srgb(L_out, ab_out)
