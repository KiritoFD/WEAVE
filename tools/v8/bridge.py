"""V8 Brownian bridge / conditional flow path sampler.

All paths use the same scaled latent protocol as D5 VAE posterior means.

    z_tau = (1-t) z0 + t z1 + 2*sigma_mid*sqrt(t(1-t))*xi
    v_tau = z1-z0 + sigma_mid*(1-2t)/sqrt(t(1-t))*xi

sigma_mid is the *noise std at tau=0.5*; sigma_mid=0 recovers straight CFM.
The path uses the raw tau (so exact endpoints hold); only the singular velocity
denominator is clamped. Training samples tau in [0.01, 0.99].
"""
from __future__ import annotations

from typing import Tuple

import torch


def sample_bridge(
    z0: torch.Tensor,
    z1: torch.Tensor,
    tau: torch.Tensor,
    sigma_mid: float = 0.0,
    noise: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_raw = tau.float().view(-1, 1, 1, 1)
    t_clamped = t_raw.clamp(1e-4, 1.0 - 1e-4)
    if noise is None:
        noise = torch.randn_like(z0)
    sqrt_eps = 2.0 * float(sigma_mid)
    root_raw = torch.sqrt((t_raw * (1.0 - t_raw)).clamp_min(0.0))
    root_clamped = torch.sqrt(t_clamped * (1.0 - t_clamped))
    z_tau = (1.0 - t_raw) * z0 + t_raw * z1 + sqrt_eps * root_raw * noise
    if sqrt_eps == 0.0:
        noise_velocity = torch.zeros_like(z0)
    else:
        # Derivative of gamma(t)=sqrt(eps*t(1-t)); singular at endpoints, clamped.
        noise_velocity = sqrt_eps * (1.0 - 2.0 * t_clamped) / (2.0 * root_clamped) * noise
    velocity = z1 - z0 + noise_velocity
    return z_tau, velocity


def sample_bridge_with_seed(
    z0: torch.Tensor,
    z1: torch.Tensor,
    tau: torch.Tensor,
    sigma_mid: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=z0.device).manual_seed(seed)
    noise = torch.randn(z0.shape, generator=g, device=z0.device, dtype=z0.dtype)
    return sample_bridge(z0, z1, tau, sigma_mid, noise)
