"""V10.1 relation features: how strokes are ORGANISED, not how much energy.

17 autograd-safe soft statistics computed on an (N,4,H,W) patch (H=W=8 or 16):
  1. channel-mean gradient direction aggregated per 2x2 quadrant  (8)
  2. pairwise cosine between quadrant direction vectors            (6)
  3. quadrant HF-energy max/min and std/mean                       (2)
  4. correlation of quadrant LL energy vs HF energy                (1)

v10_1 stroke vector v = f_feats(34) + relation_feats(17) = 51 dims (both scales).
"""
from __future__ import annotations

import torch

D_REL = 17


def _quad(p: torch.Tensor) -> torch.Tensor:
    """(N,4,H,W) -> (N,4,2,H/2,2,W/2) quadrants (row-major: tl,tr,bl,br)."""
    n, c, h, w = p.shape
    return p.reshape(n, c, 2, h // 2, 2, w // 2).permute(0, 2, 4, 1, 3, 5) \
            .reshape(n, 4, c, h // 2, w // 2)


def relation_feats(p: torch.Tensor) -> torch.Tensor:
    """(N,4,H,W) with even H,W -> (N,17)."""
    n = p.shape[0]
    q = _quad(p)                                      # (N,4,4,h,w)
    # channel-mean gradient field per quadrant
    gh = q[:, :, :, :, 1:] - q[:, :, :, :, :-1]       # (N,4,4,h,w-1)
    gv = q[:, :, :, 1:, :] - q[:, :, :, :-1, :]
    dh = gh.mean(dim=(2, 3, 4))                       # (N,4)
    dv = gv.mean(dim=(2, 3, 4))
    dirs = torch.stack([dh, dv], dim=2).reshape(n, 8)  # (N,8)
    dvec = torch.stack([dh, dv], dim=2)               # (N,4,2)
    dn = torch.nn.functional.normalize(dvec, dim=2, eps=1e-8)
    idx = torch.triu_indices(4, 4, offset=1)
    cos6 = (dn[:, idx[0]] * dn[:, idx[1]]).sum(-1)    # (N,6)
    # quadrant HF energy (residual after 3x3 box)
    blur = torch.nn.functional.avg_pool2d(
        p.reshape(-1, 4, p.shape[-2], p.shape[-1]), 3, stride=1, padding=1,
        count_include_pad=False).reshape_as(p)
    hf = p - blur
    hf_e = _quad(hf).square().mean(dim=(2, 3, 4))     # (N,4) per quadrant
    # flat quadrants give hf_e ~ 0 -> guard before ratios (unclamped max/min
    # explodes to 1e8+ and overflows the gradient)
    hf_c = hf_e.clamp_min(1e-6)
    hf_ratio = torch.stack([
        (hf_c.max(1).values / hf_c.min(1).values).clamp_max(1e3),
        (hf_e.std(1, unbiased=False) / hf_c.mean(1)).clamp_max(1e3)],
        dim=1)                                        # (N,2)
    # LL energy per quadrant and its correlation with HF energy
    ll = torch.nn.functional.avg_pool2d(
        p, 2)                                         # (N,4,H/2,W/2)
    ll_e = _quad(ll).square().mean(dim=(2, 3, 4))     # (N,4)
    a = ll_e - ll_e.mean(1, keepdim=True)
    b = hf_e - hf_e.mean(1, keepdim=True)
    corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
    corr = corr.unsqueeze(1)                          # (N,1)
    return torch.cat([dirs, cos6, hf_ratio, corr], dim=1)


def v_feats(p: torch.Tensor) -> torch.Tensor:
    """Stroke vector: f(34) + relations(17) = 51 (any even spatial size)."""
    from .features import f_feats
    return torch.cat([f_feats(p), relation_feats(p)], dim=1)
