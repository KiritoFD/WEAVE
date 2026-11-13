"""Stratified time sampling with importance weights (experiment 15).

Sampling t non-uniformly changes the loss, so the per-sample weight p(t)/q(t)
must be applied to keep the original objective in expectation.
"""
from __future__ import annotations

import torch


def stratified_t(batch: int, min_t: float, n_strata: int, gen: torch.Generator,
                 device: str, dtype=torch.float32):
    """Sample t from `n_strata` equal-width strata over [min_t, 1-min_t].

    q(t) = 1/(K * w_k) inside stratum k, and the target density is uniform on
    [min_t, 1-min_t] with density 1/W. Hence the importance weight is
    p/q = (1/W) / (1/(K w_k)) = K w_k / W. With equal-width strata this is 1,
    so the weight only differs if the strata are made unequal; the returned
    weights are nevertheless exact and are used as-is.
    """
    lo, hi = min_t, 1.0 - min_t
    W = hi - lo
    k = torch.randint(0, n_strata, (batch,), generator=gen, device=device)
    w = W / n_strata
    u = torch.rand(batch, generator=gen, device=device, dtype=dtype)
    t = lo + k.to(dtype) * w + u * w
    weight = (n_strata * w) / W
    return t.clamp(lo, hi), torch.full((batch,), float(weight), device=device, dtype=dtype)


def uniform_t(batch: int, min_t: float, gen: torch.Generator, device: str, dtype=torch.float32):
    t = torch.rand(batch, generator=gen, device=device, dtype=dtype) * (1 - 2 * min_t) + min_t
    return t, torch.ones_like(t)


def sample_time(mode: str, batch: int, min_t: float, gen: torch.Generator, device: str,
                n_strata: int = 8, dtype=torch.float32):
    if mode == "uniform":
        return uniform_t(batch, min_t, gen, device, dtype)
    if mode == "stratified":
        return stratified_t(batch, min_t, n_strata, gen, device, dtype)
    raise ValueError(f"unknown time sampling mode {mode}")
