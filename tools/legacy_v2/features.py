"""固定（无参数、可微）风格描述子：局部 Φ_l 与整图 q。

设计约束（V3-SWD）：
* 只用颜色 / 局部对比度 / 方向纹理 / 笔触，不含学习参数，不做语义嵌入；
* 生成侧必须可微（损失要反传到 latent）；
* 真实侧同一条代码路径离线缓存，保证分布可比。

局部描述子（每个 patch，14 维）：
  mean RGB(3), std RGB(3), luma mean(1), luma std(1),
  |∇luma| mean(1), 4 方向投影能量(4), |Laplacian| mean(1)

整图描述子（86 维）：
  soft 色彩直方图 3x16 = 48
  方向频谱 8 角 x 4 径向 = 32（在 128px luma 上做 FFT）
  跨尺度纹理统计 mean/std(|Laplacian|) @ 512/256/128 = 6
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

LOCAL_DIM = 14
GLOBAL_DIM = 86

# (下采样因子, patch, stride) —— 三个尺度都给出 ~63x63 的 patch 网格
LOCAL_LEVELS: tuple[tuple[int, int, int], ...] = ((1, 7, 8), (2, 5, 4), (4, 3, 2))
PATCHES_PER_LEVEL = 128  # 每张图每尺度保留的 patch 数（离线缓存用）


def _luma(x: torch.Tensor) -> torch.Tensor:
    return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]


def _fixed_kernels(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype
    ) / 8.0
    sobel_y = sobel_x.t()
    lap = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype
    )
    blur = torch.tensor(
        [[1.0, 4.0, 6.0, 4.0, 1.0],
         [4.0, 16.0, 24.0, 16.0, 4.0],
         [6.0, 24.0, 36.0, 24.0, 6.0],
         [4.0, 16.0, 24.0, 16.0, 4.0],
         [1.0, 4.0, 6.0, 4.0, 1.0]], device=device, dtype=dtype
    ) / 256.0
    return {"sobel_x": sobel_x, "sobel_y": sobel_y, "lap": lap, "blur": blur}


def _conv2d(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, k.view(1, 1, *k.shape), padding=k.shape[0] // 2)


def _grad_terms(y: torch.Tensor, kernels: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gx = _conv2d(y, kernels["sobel_x"])
    gy = _conv2d(y, kernels["sobel_y"])
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return gx, gy, mag


def local_descriptors(
    x: torch.Tensor,
    *,
    levels: tuple[tuple[int, int, int], ...] = LOCAL_LEVELS,
) -> list[torch.Tensor]:
    """x: [B,3,H,W] in ~[0,1] -> 每尺度 [B, 14, h, w] 的 patch 描述子网格。"""
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"features expect (B,3,H,W), got {tuple(x.shape)}")
    kernels = _fixed_kernels(x.device, x.dtype)
    out: list[torch.Tensor] = []
    for factor, patch, stride in levels:
        img = x if factor == 1 else F.interpolate(x, scale_factor=1.0 / factor, mode="area")
        y = _luma(img)
        mean = F.avg_pool2d(img, patch, stride)
        mean_sq = F.avg_pool2d(img * img, patch, stride)
        std = torch.sqrt(torch.clamp(mean_sq - mean * mean, min=1e-6))
        ly = F.avg_pool2d(y, patch, stride)
        ly_sq = F.avg_pool2d(y * y, patch, stride)
        lstd = torch.sqrt(torch.clamp(ly_sq - ly * ly, min=1e-6))
        gx, gy, mag = _grad_terms(y, kernels)
        gmag = F.avg_pool2d(mag, patch, stride)
        orient = []
        for a in (0.0, 0.7853981634, 1.5707963268, 2.3561944902):
            proj = (gx * torch.cos(torch.tensor(a, device=x.device, dtype=x.dtype)) +
                    gy * torch.sin(torch.tensor(a, device=x.device, dtype=x.dtype))).abs()
            orient.append(F.avg_pool2d(proj, patch, stride))
        lap = _conv2d(y, kernels["lap"]).abs()
        lapm = F.avg_pool2d(lap, patch, stride)
        desc = torch.cat([mean, std, ly, lstd, gmag, *orient, lapm], dim=1)
        out.append(desc)
    return out


def global_descriptor(x: torch.Tensor, *, bins: int = 16, spec_size: int = 128) -> torch.Tensor:
    """x: [B,3,H,W] -> [B,86] 整图描述子。"""
    b = x.shape[0]
    kernels = _fixed_kernels(x.device, x.dtype)

    # --- 色彩直方图（soft bin，可微）---
    centers = (torch.arange(bins, device=x.device, dtype=x.dtype) + 0.5) / bins
    width = 1.0 / bins
    hist = []
    for c in range(3):
        v = x[:, c].flatten(1)  # [B, H*W]
        w = torch.clamp(1.0 - torch.abs(v[:, None, :] - centers[None, :, None]) / width, min=0.0)
        hist.append(w.mean(dim=2))
    color = torch.cat(hist, dim=1)  # [B, 48]

    # --- 方向频谱（128px luma FFT，径向/角度分块）---
    y128 = _luma(F.interpolate(x, size=(spec_size, spec_size), mode="area"))[:, 0]
    spec = torch.fft.fftshift(torch.fft.fft2(y128), dim=(-2, -1)).abs()
    masks = _spectral_masks(spec_size, x.device, spec.dtype)
    spec_bins = torch.einsum("bhw,khw->bk", spec, masks)
    spec_bins = spec_bins / spec_bins.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # --- 跨尺度纹理 ---
    tex = []
    for factor in (1, 2, 4):
        img = x if factor == 1 else F.interpolate(x, scale_factor=1.0 / factor, mode="area")
        lap = _conv2d(_luma(img), kernels["lap"]).abs()
        tex.append(lap.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1))
        tex.append(lap.std(dim=(1, 2, 3), keepdim=False).unsqueeze(1))
    texture = torch.cat(tex, dim=1)  # [B, 6]

    return torch.cat([color, spec_bins, texture], dim=1)


_MASK_CACHE: dict[tuple[int, str, str], torch.Tensor] = {}


def _spectral_masks(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (size, str(device), str(dtype))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    ang_bins, rad_bins = 8, 4
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy = cx = (size - 1) / 2.0
    dy, dx = yy - cy, xx - cx
    r = torch.sqrt(dy * dy + dx * dx)
    theta = torch.atan2(dy, dx) + torch.pi  # [0, 2pi)
    rmax = r.max()
    ridx = torch.clamp((r / rmax * rad_bins).floor().long(), max=rad_bins - 1)
    aidx = torch.clamp((theta / (2 * torch.pi) * ang_bins).floor().long(), max=ang_bins - 1)
    idx = aidx * rad_bins + ridx
    masks = F.one_hot(idx.flatten(), num_classes=ang_bins * rad_bins).to(torch.float32)
    masks = masks.view(size, size, -1).permute(2, 0, 1).contiguous().to(dtype)
    # 每个块按面积归一，避免大块支配
    masks = masks / masks.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    _MASK_CACHE[key] = masks
    return masks


def fit_whitener(x: torch.Tensor, labels: torch.Tensor, num_classes: int, *, ridge: float = 1e-2) -> torch.Tensor:
    """只用真实作品拟合的类内协方差白化（类 Fisher LDA 的度量，冻结后当固定特征变换）。

    x: [N,D]（已 z-score）；labels: [N] 风格 id。返回 [D,D] 的 W，使 y = W (x - mu) 后在
    类内白、类间拉大。ridge 为相对正则，防止小特征值放大噪声。
    """
    d = x.shape[1]
    mu = x.mean(dim=0)
    xc = x - mu
    within = x.new_zeros((d, d))
    for c in range(num_classes):
        sub = xc[labels == c]
        if sub.shape[0] == 0:
            continue
        within = within + sub.transpose(0, 1) @ sub / max(sub.shape[0] - 1, 1)
    within = within / max(num_classes, 1)
    scale = torch.diagonal(within).mean().clamp_min(1e-8)
    within = within + torch.eye(d, device=x.device, dtype=x.dtype) * (ridge * scale)
    chol = torch.linalg.cholesky(within)
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    w = torch.linalg.solve_triangular(chol, eye, upper=False)  # W^T W = within^{-1}
    return w.transpose(0, 1).contiguous()


@dataclass
class FeatureStats:
    """特征库的固定白化参数（离线算一次，训练/评测共用）：
    z-score(mu/sigma) 后接类内协方差白化 W（只用真实作品拟合，冻结）。
    """

    local_mean: torch.Tensor   # [3, 14]
    local_std: torch.Tensor    # [3, 14]
    local_whiten: torch.Tensor  # [3, 14, 14]
    global_mean: torch.Tensor   # [86]
    global_std: torch.Tensor    # [86]
    global_whiten: torch.Tensor  # [86, 86]

    def to(self, device: torch.device) -> "FeatureStats":
        return FeatureStats(
            self.local_mean.to(device), self.local_std.to(device), self.local_whiten.to(device),
            self.global_mean.to(device), self.global_std.to(device), self.global_whiten.to(device),
        )


def normalize_local(desc: list[torch.Tensor], stats: FeatureStats) -> list[torch.Tensor]:
    out = []
    for i, d in enumerate(desc):
        m = stats.local_mean[i].view(1, -1, 1, 1).to(d.dtype)
        s = stats.local_std[i].view(1, -1, 1, 1).to(d.dtype)
        w = stats.local_whiten[i].to(d.dtype)
        out.append(torch.einsum("bchw,dc->bdhw", (d - m) / s, w))
    return out


def encode_style(encoder, x: torch.Tensor, *, size: int = 128) -> torch.Tensor:
    """冻结 Style Encoder 的 128-d 嵌入（区域平均下采样到 128px）。"""
    xin = F.interpolate(x, size=(size, size), mode="area") if x.shape[-1] != size else x
    emb, _logits = encoder(xin)
    return emb


def normalize_global(q: torch.Tensor, stats: FeatureStats) -> torch.Tensor:
    z = (q - stats.global_mean.to(q.dtype)) / stats.global_std.to(q.dtype)
    return z @ stats.global_whiten.to(q.dtype).transpose(0, 1)


__all__ = [
    "LOCAL_DIM",
    "GLOBAL_DIM",
    "LOCAL_LEVELS",
    "PATCHES_PER_LEVEL",
    "FeatureStats",
    "local_descriptors",
    "global_descriptor",
    "encode_style",
    "fit_whitener",
    "normalize_local",
    "normalize_global",
]
