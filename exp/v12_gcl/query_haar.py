"""G1 Query: multi-scale Haar structure descriptor of a latent (zero params).

Q(z) = [LL16, LL32, dLL16/dx, dLL16/dy, dLL32/dx, dLL32/dy]  (flattened, ~15k dims)
Loss: SmoothL1(Q(z), stopgrad(Q(z_src))) reduced per sample.

Pre-registered (plan §3-G1): candidate Q compared on the ACTUAL edited latent;
source anchor stop-gradient; SmoothL1 keeps coarse/fine terms and does not lock
high-frequency stroke energy (L1-like subgradient at zero).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn


def haar_query(z: torch.Tensor) -> torch.Tensor:
    """(B,4,64,64) -> (B, D) multi-scale Haar structure query."""
    ll32 = Fn.avg_pool2d(z, 2)                    # (B,4,32,32)
    ll16 = Fn.avg_pool2d(z, 4)                    # (B,4,16,16)
    parts = [ll16.flatten(1), ll32.flatten(1)]
    for ll in (ll16, ll32):
        gx = ll[..., :, 1:] - ll[..., :, :-1]     # (B,4,H,W-1)
        gy = ll[..., 1:, :] - ll[..., :-1, :]
        parts.append(gx.flatten(1))
        parts.append(gy.flatten(1))
    return torch.cat(parts, 1)


def haar_gcl_loss(z: torch.Tensor, z_src: torch.Tensor) -> torch.Tensor:
    """Per-sample SmoothL1 between Q(z) and stopgrad(Q(z_src)) -> (B,)."""
    q = haar_query(z)
    with torch.no_grad():
        qs = haar_query(z_src.detach())
    return Fn.smooth_l1_loss(q, qs, reduction="none").mean(1)
