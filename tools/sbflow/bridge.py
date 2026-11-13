"""Brownian bridge one-time marginals and bidirectional drift targets.

Epsilon is the Brownian diffusion variance, not a mid-point standard deviation:
  std(X_t | X_0, X_1) = sqrt(epsilon*t*(1-t)).
These one-time samples are not a Brownian path sample across multiple t.
"""
import torch


def bridge_sample(z0, z1, t, epsilon, noise=None):
    if epsilon <= 0:
        raise ValueError('Bridge reproduction requires epsilon > 0')
    t = t.to(device=z0.device, dtype=z0.dtype).reshape(-1, *([1]*(z0.ndim-1)))
    if (t <= 0).any() or (t >= 1).any():
        raise ValueError('Bridge regression requires t strictly between 0 and 1')
    xi = torch.randn_like(z0) if noise is None else noise
    xt = (1-t)*z0 + t*z1 + (epsilon*t*(1-t)).sqrt()*xi
    yplus = (z1-xt)/(1-t)
    yminus = (z0-xt)/t
    return xt, yplus, yminus


def exact_conditional_flow(z0, z1, t, epsilon, xi):
    """Pathwise derivative of a single-time Gaussian interpolation; NOT SDE drift."""
    t = t.reshape(-1, *([1]*(z0.ndim-1)))
    return (z1-z0) + (epsilon**0.5)*(1-2*t)/(2*(t*(1-t)).sqrt())*xi
