"""Shared de-styled structure, luminance-chrominance projection, and Oklab color utilities."""
from __future__ import annotations

import math
from PIL import Image
import torch
import torch.nn.functional as F


def rgb_to_gray(rgb01: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luminance Y: (B, 3, H, W) in [0, 1] -> (B, 1, H, W)."""
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)


def gaussian_blur_3x3(x: torch.Tensor) -> torch.Tensor:
    """Depthwise reflection-padded 3x3 binomial Gaussian filter."""
    c = x.shape[1]
    k = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=x.device, dtype=x.dtype,
    ).view(1, 1, 3, 3) / 16.0
    k_c = k.repeat(c, 1, 1, 1)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k_c, groups=c)


def local_contrast_norm(y: torch.Tensor, kernel: int = 5, eps: float = 0.03) -> torch.Tensor:
    """Gradient-safe zero-mean local-RMS-equalized structure map in [-1, 1]."""
    pad = kernel // 2
    yp = F.pad(y, (pad, pad, pad, pad), mode="reflect")
    mean = F.avg_pool2d(yp, kernel, stride=1)
    diff = y - mean
    dp = F.pad(diff.square(), (pad, pad, pad, pad), mode="reflect")
    std = F.avg_pool2d(dp, kernel, stride=1).clamp_min(1e-8).sqrt()
    return (diff / (std + eps)).clamp(-3.0, 3.0) / 3.0


def y64_to_structure_4ch(y64: torch.Tensor) -> torch.Tensor:
    """Convert 64x64 1-channel luminance map to 4-ch [LCN_5, LCN_13, Sobel95, LapRMS]."""
    y64 = y64.float()
    lcn_fine = local_contrast_norm(y64, kernel=5, eps=0.03)
    lcn_coarse = local_contrast_norm(y64, kernel=13, eps=0.04)

    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=y64.device, dtype=y64.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    yp = F.pad(y64, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(yp, kx)
    gy = F.conv2d(yp, ky)
    mag = (gx.square() + gy.square() + 1e-8).sqrt()
    q95 = torch.quantile(mag.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1).clamp_min(0.05)
    edge_norm = (mag / q95).clamp(0.0, 1.5) / 1.5 * 2.0 - 1.0

    klap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                        device=y64.device, dtype=y64.dtype).view(1, 1, 3, 3)
    lap = F.conv2d(yp, klap)
    lap_rms = lap.flatten(1).square().mean(1).clamp_min(1e-8).sqrt().view(-1, 1, 1, 1).clamp_min(0.02)
    lap_norm = (lap / (3.0 * lap_rms)).clamp(-1.0, 1.0)
    return torch.cat([lcn_fine, lcn_coarse, edge_norm, lap_norm], dim=1)


def rgb_to_structure_64(rgb01: torch.Tensor) -> torch.Tensor:
    """Extract palette-free 4-ch 64x64 structure from 512x512 RGB [0, 1]."""
    y512 = rgb_to_gray(rgb01.float())
    y128 = F.avg_pool2d(y512, kernel_size=4, stride=4)
    y64 = F.avg_pool2d(gaussian_blur_3x3(y128), kernel_size=2, stride=2)
    return y64_to_structure_4ch(y64)


def latent_to_structure_64(z: torch.Tensor, w_luma: torch.Tensor, b_luma: torch.Tensor) -> torch.Tensor:
    """Zero-decode 0.05ms latent-domain 4-ch luminance structure map from 64x64 latent z!"""
    w = w_luma.to(device=z.device, dtype=torch.float32).view(1, 4, 1, 1)
    b = b_luma.to(device=z.device, dtype=torch.float32)
    y64 = ((z.float() * w).sum(dim=1, keepdim=True) + b).clamp(0.0, 1.0)
    return y64_to_structure_4ch(y64)


@torch.no_grad()
def extract_whitened_dino_layout(dino_model, rgb01: torch.Tensor, batch: int = 16, device: str = "cuda") -> torch.Tensor:
    """Extract zero-mean spherical grayscale DINOv2 16x16 patch layout (B, 384, 16, 16)."""
    from src.utils._local_eval_refs import encode_dino

    y = rgb_to_gray(rgb01.float())
    gray3 = y.repeat(1, 3, 1, 1)
    u8 = (gray3.cpu() * 255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    imgs = [Image.fromarray(arr) for arr in u8]
    _, patch = encode_dino(dino_model, imgs, batch=batch, device=device)
    p = patch.float().transpose(1, 2).reshape(len(imgs), 384, 16, 16)
    mean = p.mean(dim=(2, 3), keepdim=True)
    std = p.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
    p_whiten = (p - mean) / std
    p_norm = p_whiten / p_whiten.norm(dim=1, keepdim=True).clamp_min(1e-6) * math.sqrt(384.0)
    return p_norm - p_norm.mean(dim=(2, 3), keepdim=True)


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB [0, 1] -> perceptual Oklab (B, 3, H, W)."""
    x = rgb.clamp(0.0, 1.0).float()
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(1e-6) ** 2.4)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = l.clamp_min(1e-6).pow(1.0 / 3.0)
    m_ = m.clamp_min(1e-6).pow(1.0 / 3.0)
    s_ = s.clamp_min(1e-6).pow(1.0 / 3.0)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.cat([L, A, B], dim=1)


def extract_color_fiber_27d(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable 27-D Oklab palette moments, quantiles, and 3x3 covariance."""
    lab = rgb_to_oklab(rgb)
    B = lab.shape[0]
    flat = lab.flatten(2)
    mean = flat.mean(dim=-1)
    diff = flat - mean.unsqueeze(-1)
    std = (diff.square().mean(dim=-1) + 1e-6).sqrt()
    cov = torch.bmm(diff, diff.transpose(1, 2)) / max(flat.shape[-1] - 1, 1)
    triu_idx = torch.triu_indices(3, 3, device=rgb.device)
    cov6 = cov[:, triu_idx[0], triu_idx[1]]
    sub = flat[:, :, ::max(1, flat.shape[-1] // 4096)]
    qs = torch.quantile(sub, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=rgb.device), dim=-1)
    q15 = qs.permute(1, 0, 2).reshape(B, 15)
    return torch.cat([mean, std, cov6, q15], dim=-1)


def dwt2_haar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthonormal single-level 2D Haar DWT: (B, C, H, W) -> (LL, LH, HL, HH) at (H/2, W/2)."""
    orig_dtype = x.dtype
    xf = x.float()
    if xf.shape[2] % 2 != 0:
        xf = F.pad(xf, (0, 0, 0, 1), mode="replicate")
    if xf.shape[3] % 2 != 0:
        xf = F.pad(xf, (0, 1, 0, 0), mode="replicate")
    B, C, H, W = xf.shape
    xr = xf.reshape(B, C, H // 2, 2, W // 2, 2)
    a, b = xr[:, :, :, 0, :, 0], xr[:, :, :, 0, :, 1]
    c, d = xr[:, :, :, 1, :, 0], xr[:, :, :, 1, :, 1]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll.to(orig_dtype), lh.to(orig_dtype), hl.to(orig_dtype), hh.to(orig_dtype)


def idwt2_haar(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Exact inverse orthonormal 2D Haar IDWT: 4x (B, C, H/2, W/2) -> (B, C, H, W)."""
    orig_dtype = ll.dtype
    ll_f, lh_f, hl_f, hh_f = ll.float(), lh.float(), hl.float(), hh.float()
    a = (ll_f + lh_f + hl_f + hh_f) * 0.5
    b = (ll_f + lh_f - hl_f - hh_f) * 0.5
    c = (ll_f - lh_f + hl_f - hh_f) * 0.5
    d = (ll_f - lh_f - hl_f + hh_f) * 0.5
    B, C, H2, W2 = ll_f.shape
    top = torch.stack([a, b], dim=-1)
    bot = torch.stack([c, d], dim=-1)
    out = torch.stack([top, bot], dim=-2).reshape(B, C, H2 * 2, W2 * 2)
    return out.to(orig_dtype)

