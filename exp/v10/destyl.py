"""Latent-domain auto-destylization operators (pre-registered ladder, plan §5-A).

Each operator maps a real painting latent z to a degraded z^- whose known
recovery target is z itself. All run under no_grad; z^- becomes an autograd
leaf at the call site. Seven operators (3-ladder + 4 single), fixed order:

  att03/att05/att07 : HF (3x3 box residual) attenuation, alpha in {0.3,0.5,0.7}
  dirx05 / diry05   : anisotropic smoothing along x / y (kernel 5, alpha 0.5)
  wave05            : Haar HF-band weakening (LL kept), alpha 0.5
  color03           : partial per-channel color normalization toward the
                      train-split style-pool statistics, beta 0.3
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

LADDER = ["att03", "att05", "att07", "dirx05", "diry05", "wave05", "color03"]


def _box3(z: torch.Tensor) -> torch.Tensor:
    return Fn.avg_pool2d(z, 3, stride=1, padding=1, count_include_pad=False)


def _box_axis(z: torch.Tensor, axis: int, k: int = 5) -> torch.Tensor:
    if axis == 0:
        return Fn.avg_pool2d(z, (1, k), stride=(1, 1),
                             padding=(0, k // 2), count_include_pad=False)
    return Fn.avg_pool2d(z, (k, 1), stride=(1, 1),
                         padding=(k // 2, 0), count_include_pad=False)


def _haar_ll_up(z: torch.Tensor) -> torch.Tensor:
    ll = Fn.avg_pool2d(z, 2)
    return Fn.interpolate(ll, size=z.shape[-2:], mode="nearest")


@torch.no_grad()
def degrade(z: torch.Tensor, op: str,
            pool_mean: torch.Tensor | None = None,
            pool_std: torch.Tensor | None = None) -> torch.Tensor:
    """z (B,4,64,64) -> z^- (same shape). op in LADDER."""
    if op.startswith("att"):
        a = float(op[3:]) / 10.0
        return z - a * (z - _box3(z))
    if op == "dirx05":
        return z - 0.5 * (z - _box_axis(z, 0))
    if op == "diry05":
        return z - 0.5 * (z - _box_axis(z, 1))
    if op == "wave05":
        ll = _haar_ll_up(z)
        return ll + 0.5 * (z - ll)
    if op == "color03":
        assert pool_mean is not None and pool_std is not None
        m = z.mean(dim=(0, 2, 3), keepdim=True)
        s = z.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
        # shrink z's per-channel affine toward the style pool's
        m2 = m + 0.3 * (pool_mean.view(1, -1, 1, 1).to(z.device) - m)
        s2 = s + 0.3 * (pool_std.view(1, -1, 1, 1).to(z.device) - s)
        return (z - m) / s * s2 + m2
    raise ValueError(op)
