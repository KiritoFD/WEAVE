"""可直接计算、可微的分布距离：sliced Wasserstein（1D 投影）与多核 RBF-MMD。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _quantile_rows(sorted_x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """sorted_x: [N,D] 已按 dim=0 排序；返回 [Q,D] 的分位数（线性插值）。"""
    n = sorted_x.shape[0]
    if n == 1:
        return sorted_x.expand(q.numel(), -1)
    pos = q * (n - 1)
    lo = torch.floor(pos).long().clamp(0, n - 1)
    hi = torch.ceil(pos).long().clamp(0, n - 1)
    frac = (pos - lo.to(pos.dtype)).unsqueeze(1)
    return sorted_x[lo] * (1.0 - frac) + sorted_x[hi] * frac


def swd2(a: torch.Tensor, b: torch.Tensor, *, n_quantiles: int = 128) -> torch.Tensor:
    """Sliced-Wasserstein-2^2（坐标轴为投影方向）。

    a: [N,D]（生成侧，需梯度），b: [M,D]（真实侧，无梯度）。
    返回标量；按维度求和以保持与 D 无关的平均步长。
    """
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"swd2 expects [N,D]/[M,D] with same D, got {tuple(a.shape)} vs {tuple(b.shape)}")
    q = torch.linspace(0.0, 1.0, n_quantiles, device=a.device, dtype=a.dtype)
    a_sorted = torch.sort(a, dim=0).values
    b_sorted = torch.sort(b.detach().float(), dim=0).values.to(a_sorted.dtype)
    a_q = _quantile_rows(a_sorted, q)
    b_q = _quantile_rows(b_sorted, q)
    return ((a_q - b_q) ** 2).mean(dim=0).sum()


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    d2 = torch.cdist(x, y) ** 2
    return torch.exp(-d2 / (2.0 * sigma * sigma))


def mmd2_rbf(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    max_real: int = 64,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """多核 RBF-MMD^2（有偏估计，保证非负、可微）。

    sigma0 用中位数启发式在两集合上估计（detach），核宽取 sigma0 * multipliers。
    """
    b = b.detach()
    if b.shape[0] > max_real:
        idx = torch.randperm(b.shape[0], generator=gen, device=b.device)[:max_real]
        b = b.index_select(0, idx)
    a_f = a.float()
    b_f = b.float()
    with torch.no_grad():
        both = torch.cat([a_f, b_f], dim=0)
        d = torch.cdist(both, both)
        sigma0 = torch.median(d[d > 0]).clamp_min(1e-3)
    total = a.new_zeros(())
    for m in multipliers:
        sigma = float(sigma0) * float(m)
        kaa = _rbf_kernel(a_f, a_f, sigma)
        kbb = _rbf_kernel(b_f, b_f, sigma)
        kab = _rbf_kernel(a_f, b_f, sigma)
        total = total + (kaa.mean() + kbb.mean() - 2.0 * kab.mean())
    return total / float(len(multipliers))


def content_structure_loss(x_hat: torch.Tensor, x: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    """||SS(ŷ)-SS(x)||_1 + alpha*||∇B(Y(ŷ))-∇B(Y(x))||_1."""
    from .features import _conv2d, _fixed_kernels, _luma

    ss = x.new_zeros(())
    for grid in (8, 16):
        pool = x.shape[-1] // grid
        pa = F.avg_pool2d(x_hat, pool)
        pb = F.avg_pool2d(x.detach(), pool)
        na = F.normalize(pa.flatten(2), dim=1)
        nb = F.normalize(pb.flatten(2), dim=1)
        sa = torch.bmm(na.transpose(1, 2), na)
        sb = torch.bmm(nb.transpose(1, 2), nb)
        ss = ss + (sa - sb).abs().mean()
    kernels = _fixed_kernels(x_hat.device, x_hat.dtype)
    ya = _conv2d(_luma(x_hat), kernels["blur"])
    yb = _conv2d(_luma(x.detach()), kernels["blur"])
    mag_a = torch.sqrt(_conv2d(ya, kernels["sobel_x"]) ** 2 + _conv2d(ya, kernels["sobel_y"]) ** 2 + 1e-6)
    mag_b = torch.sqrt(_conv2d(yb, kernels["sobel_x"]) ** 2 + _conv2d(yb, kernels["sobel_y"]) ** 2 + 1e-6)
    grad = (mag_a - mag_b).abs().mean()
    return ss + alpha * grad


__all__ = ["swd2", "mmd2_rbf", "content_structure_loss"]
