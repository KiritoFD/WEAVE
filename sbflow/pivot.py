"""Shared Gaussian pivot (group F, experiments 32-35).

Instead of transporting directly between art domains, train one conditional flow

    F_s : N(0, I) -> p_s

for every style, then compose

    T_{i->j} = F_j o F_i^{-1}

One model, five conditional generators, twenty directed transfers.

Implementation choice
---------------------
The Gaussian pivot is added as an extra "style" whose latent distribution is
standard normal. That way the existing ten-pair machinery handles it unchanged:
pairs become {(pivot, s)} for every real style s, and a conversion between two real
styles is the two-step composition pivot->j after i->pivot.

Caveat from the doc: cycle consistency is automatic when the maps are numerically
invertible, but that says nothing about semantic content preservation.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from .preconditioning import StyleStats


def add_pivot_style(data, n_pivot: int = 4096, seed: int = 0):
    """Append a standard-normal 'style' to a StyleData object and restrict pairs to it."""
    channels, H, W = data.z[0].shape[1:]
    g = torch.Generator().manual_seed(seed)
    z_p = torch.randn(n_pivot, channels, H, W, generator=g).to(data.z[0].device)
    thumb_p = torch.full((n_pivot, 3, 32, 32), 0.5, device=data.thumb[0].device)

    data.z.append(z_p)
    data.thumb.append(thumb_p)
    data.styles = list(data.styles) + ["__gaussian_pivot__"]
    data.counts = list(data.counts) + [n_pivot]

    st = data.stats
    mean_p = torch.zeros(channels)
    var_p = torch.ones(channels)
    data.stats = StyleStats(
        global_mean=st.global_mean, global_std=st.global_std,
        style_mean=torch.cat([st.style_mean, mean_p[None]], 0),
        style_var=torch.cat([st.style_var, var_p[None]], 0),
    )
    return data


def pivot_pairs(num_real: int) -> List[Tuple[int, int]]:
    """Only (pivot, style) pairs, canonical order puts the real style first."""
    p = num_real
    return [(s, p) for s in range(num_real)]


def invert_pivot_step(model, z, style: int, pivot: int, stats: StyleStats, eps: float,
                      steps: int, method: str, t_floor: float, objective: str) -> torch.Tensor:
    """Map a real style latent back to the Gaussian pivot (i.e. F_i^{-1})."""
    from .infer5 import solve_transfer

    return solve_transfer(model, z, style, pivot, stats, eps, steps, method, t_floor, objective)


def pivot_transfer(model, z, src_style: int, tgt_style: int, pivot: int,
                   stats: StyleStats, eps: float, steps: int = 4, method: str = "heun",
                   t_floor: float = 0.002, objective: str = "bidir") -> torch.Tensor:
    """T_{i->j} = F_j o F_i^{-1} through the shared Gaussian pivot."""
    from .infer5 import solve_transfer

    mid = invert_pivot_step(model, z, src_style, pivot, stats, eps, steps, method,
                            t_floor, objective)
    return solve_transfer(model, mid, pivot, tgt_style, stats, eps, steps, method,
                          t_floor, objective)
