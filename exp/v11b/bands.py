"""Orthogonal 2-level Haar DWT for V11-B wavelet stroke flow.

x [B,3,H,W] -> LL2 [B,3,H/4,W/4], H2 [B,9,H/4,W/4], H1 [B,9,H/2,W/2].
H1 = level-1 detail (fine strokes), H2 = level-2 detail (mid strokes).
All ops are exactly invertible up to fp error.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def haar2d(x: torch.Tensor):
    """One-level orthogonal Haar DWT on last two dims.

    x [...,H,W] -> (ll [...,H/2,W/2], hd [...,3,H/2,W/2]) with detail
    channel order (horizontal, vertical, diagonal).
    """
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError("haar2d requires even H,W")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    s = a + b + c + d
    ll = s * 0.5
    hd = torch.stack(
        [(a + b - c - d) * 0.5,
         (a - b + c - d) * 0.5,
         (a - b - c + d) * 0.5],
        dim=-3)
    return ll, hd


def ihaar2d(ll: torch.Tensor, hd: torch.Tensor) -> torch.Tensor:
    """Exact inverse of haar2d."""
    hh, hv, hdg = hd[..., 0, :, :], hd[..., 1, :, :], hd[..., 2, :, :]
    # verified inverse: 2a = ll+hh+hv+hd etc.
    a = (ll + hh + hv + hdg) * 0.5
    b = (ll + hh - hv - hdg) * 0.5
    c = (ll - hh + hv - hdg) * 0.5
    d = (ll - hh - hv + hdg) * 0.5
    h, w = ll.shape[-2], ll.shape[-1]
    x = torch.empty(*ll.shape[:-2], h * 2, w * 2,
                    dtype=torch.promote_types(ll.dtype, hd.dtype),
                    device=ll.device)
    x[..., 0::2, 0::2] = a
    x[..., 0::2, 1::2] = b
    x[..., 1::2, 0::2] = c
    x[..., 1::2, 1::2] = d
    return x


def dwt2(x: torch.Tensor):
    """Two-level Haar on [B,C,H,W] (H,W divisible by 4)."""
    if x.shape[-1] % 4 or x.shape[-2] % 4:
        raise ValueError("dwt2 requires H,W divisible by 4")
    ll1, h1 = haar2d(x)
    ll2, h2 = haar2d(ll1)
    # merge (C, band) -> C*3 channels: [B,C,3,h,w] -> [B,C*3,h,w]
    return {"LL2": ll2, "H2": h2.flatten(1, 2), "H1": h1.flatten(1, 2),
            "LL1": ll1}


def _unflatten_hd(h: torch.Tensor):
    """[B,C*3,h,w] -> [B,C,3,h,w] for ihaar2d."""
    B, ch, hh, ww = h.shape
    return h.view(B, ch // 3, 3, hh, ww)


def idwt2(ll2: torch.Tensor, h2: torch.Tensor, h1: torch.Tensor):
    ll1 = ihaar2d(ll2, _unflatten_hd(h2))
    return ihaar2d(ll1, _unflatten_hd(h1))


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude. x [B,C,H,W] -> [B,1,H,W] fp32."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=torch.float32)
    ky = kx.t().contiguous()
    xc = x.float()
    if xc.dim() == 3:
        xc = xc.unsqueeze(0)
    c = xc.shape[1]
    # Sobel on the channel-mean luminance-like signal
    g = xc.mean(1, keepdim=True)
    gx = F.conv2d(g, kx.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(g, ky.view(1, 1, 3, 3), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def band_rms_clip(delta: torch.Tensor, radius: float) -> torch.Tensor:
    """Per-sample clip: scale band update so its RMS <= radius."""
    rms = delta.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
    scale = (radius / rms).clamp(max=1.0)
    return delta * scale.view(-1, *([1] * (delta.dim() - 1)))
