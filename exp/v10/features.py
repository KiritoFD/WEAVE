"""Patch features for V10: content structure q, style features f, h_psi predictor.

All features are autograd-safe (soft statistics only, no hard binning).
A latent (B,4,64,64) is split into 64 non-overlapping 8x8 patches.

  q (D=84): LL (2x2 avg -> 4ch*4*4=64) + LL grad RMS dh/dv/dd (12)
            + LL per-ch mean (4) + per-ch edge energy (4)
  f (D=34): color: per-ch mean (4) / std (4) / pairwise corr (6)
            texture: dh/dv/dd RMS per-ch (12) + blur-residual std per-ch (4)
                     + Laplacian RMS per-ch (4)

  v = f - h_psi(q)  (content-predictable part removed; h_psi shared across styles)

Pre-registered in docs/v10/WEAVE_V10_CC_SCORE_4070LAPTOP_12H.md section 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

D_Q = 84
D_F = 34
# column blocks of f (for the color/texture shortcut audit, plan section 5-B)
F_COLOR = (0, 14)     # mean 4 + std 4 + corr 6
F_TEXTURE = (14, 34)  # dir RMS 12 + blur-res 4 + laplacian 4


def patchify(z: torch.Tensor) -> torch.Tensor:
    """(B,4,64,64) -> (B*64,4,8,8); patches keep a fixed raster order (row-major)."""
    b = z.shape[0]
    return z.reshape(b, 4, 8, 8, 8, 8).permute(0, 2, 4, 1, 3, 5) \
            .reshape(b * 64, 4, 8, 8)


def _rms(t: torch.Tensor, dim: int | tuple) -> torch.Tensor:
    # clamp BEFORE sqrt: an exactly-constant region (flat colour fields!) has
    # zero variance and sqrt(0) would give an inf/NaN backward.
    return t.square().mean(dim=dim).clamp_min(1e-12).sqrt()


def _std(t: torch.Tensor, dim: int | tuple) -> torch.Tensor:
    """Biased std over spatial dims of an (N,C,H,W) tensor -> (N,C).

    Backward-safe at zero variance (sqrt is applied to a clamped variance).
    """
    e1 = t.mean(dim=dim, keepdim=True)
    e2 = t.square().mean(dim=dim, keepdim=True)
    return (e2 - e1.square()).clamp_min(1e-12).sqrt().reshape(t.shape[0], -1)


def q_feats(p: torch.Tensor) -> torch.Tensor:
    """(N,4,8,8) -> (N,84) content-structure descriptors."""
    ll = F_avg2(p)                                    # (N,4,4,4)
    dh = ll[:, :, :, 1:] - ll[:, :, :, :-1]
    dv = ll[:, :, 1:, :] - ll[:, :, :-1, :]
    dd = ll[:, :, 1:, 1:] - ll[:, :, :-1, :-1]
    ll_grad = torch.cat([
        _rms(dh, dim=(2, 3)), _rms(dv, dim=(2, 3)), _rms(dd, dim=(2, 3))], 1)
    ll_mean = ll.mean(dim=(2, 3))
    # per-channel edge energy on the raw patch (scalar per channel)
    dh8 = p[:, :, :, 1:] - p[:, :, :, :-1]
    dv8 = p[:, :, 1:, :] - p[:, :, :-1, :]
    edge = torch.sqrt(dh8.square().mean(dim=(2, 3))
                      + dv8.square().mean(dim=(2, 3)) + 1e-12)
    return torch.cat([ll.flatten(1), ll_grad, ll_mean, edge], 1)


def f_feats(p: torch.Tensor) -> torch.Tensor:
    """(N,4,8,8) -> (N,34) style descriptors (color 14 + texture 20)."""
    mean = p.mean(dim=(2, 3))                          # (N,4)
    std = _std(p, (2, 3))                              # (N,4) backward-safe
    mu = mean - mean.mean(1, keepdim=True)
    sd = std.clamp_min(1e-6)
    zc = (p - mean[:, :, None, None]) / sd[:, :, None, None]
    # pairwise channel correlation on the patch grid
    flat = zc.flatten(2)                               # (N,4,64)
    corr = (flat @ flat.transpose(1, 2) / flat.shape[-1])
    idx = torch.triu_indices(4, 4, offset=1)
    corr6 = corr[:, idx[0], idx[1]]
    dh = p[:, :, :, 1:] - p[:, :, :, :-1]
    dv = p[:, :, 1:, :] - p[:, :, :-1, :]
    dd = p[:, :, 1:, 1:] - p[:, :, :-1, :-1]
    dir_rms = torch.cat([_rms(dh, (2, 3)), _rms(dv, (2, 3)),
                         _rms(dd, (2, 3))], 1)
    blur = F_avg3(p)
    res = p - blur
    blur_res = _std(res, (2, 3))
    c = p[:, :, 1:-1, 1:-1]
    lap = c * 4 - (p[:, :, :-2, 1:-1] + p[:, :, 2:, 1:-1]
                   + p[:, :, 1:-1, :-2] + p[:, :, 1:-1, 2:])
    lap_rms = _rms(lap, (2, 3))
    return torch.cat([mean, std, corr6, dir_rms, blur_res, lap_rms], 1)


def F_avg2(p: torch.Tensor) -> torch.Tensor:
    return nn.functional.avg_pool2d(p, 2)


def F_avg3(p: torch.Tensor) -> torch.Tensor:
    return nn.functional.avg_pool2d(p, 3, stride=1, padding=1, count_include_pad=False)


class HPsi(nn.Module):
    """Shared content predictor: q -> the content-explainable part of f."""

    def __init__(self, d_q: int = D_Q, d_f: int = D_F, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_q, hidden), nn.SiLU(), nn.Linear(hidden, d_f))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q)


def feats_from_latents(z: torch.Tensor, compute_q: bool = True):
    """(B,4,64,64) -> dict with patches/q/f tensors ((B*64, D) each)."""
    p = patchify(z)
    out = {"patches": p, "f": f_feats(p)}
    if compute_q:
        out["q"] = q_feats(p)
    return out
