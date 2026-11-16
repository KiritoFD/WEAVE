"""V17 students S0-S8: residual adapters over the frozen V15-96 base.

Identity contract: every student's delta is EXACTLY zero at initialisation
(V16 adapter is zero-init; custom classes zero their output conv), and no
frozen-base gradient is ever enabled.

S0  V16 EditResidualAdapter replay (L2 + low-freq + reg), no mask/DINO.
S1  S0 trunk + where-mask head; loss cosine + BCE + light MSE (no DINO).
S2  ContextAdapter + DINO patch-token context (S1 losses).
S3  S2 with five-class centered conditioning u_t - mean_k u_k (class axis
    only — sums to zero across classes, NOT spatially orthogonal).
S4  S2 with u_t - u_src conditioning (same-style == exact zero).
S5  S3 + amplitude head (Phase II bounded soft norm).
S6  S4 + amplitude head (Phase II bounded soft norm).
S7  S6 architecture trained on the T4 teacher (only teacher-upgraded arm).
S8  S6 structure with Phase II HARD gate stopgrad(rho)>0.6 (ablation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conditioning_offset(mode: str, tid: torch.Tensor, src_label: torch.Tensor,
                        E: nn.Parameter):
    """Class-axis conditioning deltas from the label embedding table.
    centered: u_t - mean_k u_k (zero-mean over the five classes).
    target_source: u_t - u_src; same-style gives the exact zero vector."""
    if mode == 'plain':
        return 0.
    if mode == 'centered':
        mean = E[:5].mean(0, keepdim=True)
        return E[tid] - mean
    if mode == 'target_source':
        return E[tid] - E[src_label]
    raise ValueError(mode)


class S1Adapter(nn.Module):
    """V16 trunk + where-mask head.  Delta path untouched (zero-init identity)."""

    def __init__(self, width=64, blocks=3):
        super().__init__()
        from src.model.edit_adapter import EditResidualAdapter
        self.inner = EditResidualAdapter(width, blocks)
        self.where = nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, base, content, tid, dino_tokens=None, src_label=None):
        delta = self.inner(base, content, tid)
        return delta, None, self.where(torch.cat([base, content], 1))


class ContextAdapter(nn.Module):
    """S2-S8: DINO-token context, conditioning mode, optional amplitude/where."""

    def __init__(self, width=64, blocks=3, mode='plain', amplitude=False,
                 where=True):
        super().__init__()
        self.mode = mode
        self.stem = nn.Sequential(nn.Conv2d(8, width, 3, padding=1),
                                  nn.SiLU(True))
        self.dino_proj = nn.Sequential(nn.Linear(384, width), nn.SiLU(True),
                                       nn.Linear(width, width))
        self.label_embed = nn.Embedding(6, width)      # row 5 = uncond slot
        self.film = nn.Linear(width, width * 2)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, groups=width),
                nn.SiLU(True),
                nn.Conv2d(width, width, 1), nn.SiLU(True),
                nn.Conv2d(width, width, 3, padding=1), nn.SiLU(True)))
        self.out = nn.Conv2d(width, 4, 3, padding=1)
        self.amplitude = nn.Linear(384, 1) if amplitude else None
        self.where = nn.Conv2d(width, 1, 1) if where else None
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)                  # identity at init

    def forward(self, base, content, tid, dino_tokens=None, src_label=None):
        x = self.stem(torch.cat([base, content], 1))
        ctx = dino_tokens.mean((2, 3)) if dino_tokens is not None else \
            torch.zeros(len(base), x.shape[1], device=base.device)
        tok = dino_tokens if dino_tokens is not None else \
            torch.zeros(len(base), 384, 16, 16, device=base.device)
        cond = self.dino_proj(ctx) + self.label_embed(tid.clamp(0, 5)) + \
            conditioning_offset(self.mode, tid.clamp(0, 5),
                                src_label if src_label is not None else tid,
                                self.label_embed.weight)
        tok_map = F.interpolate(tok.mean(1, keepdim=True), size=(64, 64),
                                mode='bilinear', align_corners=False)
        x = x + tok_map                                   # token energy map
        g, b = self.film(cond).chunk(2, dim=1)
        x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
        for blk in self.blocks:
            x = x + blk(x)
        delta = self.out(x)
        norm_scale = (self.amplitude(ctx).squeeze(-1)
                      if self.amplitude is not None else None)
        mask_logits = self.where(x) if self.where is not None else None
        return delta, norm_scale, mask_logits


def build_student(kind: str):
    if kind == 'S0':
        from src.model.edit_adapter import EditResidualAdapter
        return EditResidualAdapter()
    if kind == 'S1':
        return S1Adapter()
    if kind in ('S2', 'S3', 'S4'):
        mode = {'S2': 'plain', 'S3': 'centered', 'S4': 'target_source'}[kind]
        return ContextAdapter(mode=mode, amplitude=False)
    if kind in ('S5', 'S6', 'S7', 'S8'):
        mode = 'centered' if kind == 'S5' else 'target_source'
        return ContextAdapter(mode=mode, amplitude=True)
    raise ValueError(kind)
