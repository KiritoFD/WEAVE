"""One-pass analytical residual editor for PRISM-1-Unified."""

import torch
import torch.nn.functional as F
from torch import nn

from exp.v18.ccf_model import BottleneckAttention, sinusoidal_time_emb
from exp.v19.fiber_duality import apply_latent_chroma_shift


class RoutedBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(16, width, affine=False)
        self.norm2 = nn.GroupNorm(16, width, affine=False)
        self.condition = nn.Linear(192, 4 * width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)

    def forward(self, x, u, route):
        scale1, bias1, scale2, bias2 = self.condition(u)[:, :, None, None].chunk(4, 1)
        gain, bias = route.chunk(2, 1)
        h = self.conv1(F.silu(self.norm1(x) * (1 + scale1 + gain) + bias1 + bias))
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale2) + bias2))
        return x + h


class UnifiedEditor(nn.Module):
    def __init__(self, stats_dim, width=192):
        super().__init__()
        self.stats = nn.Linear(stats_dim, 64)
        self.cond = nn.Sequential(nn.Linear(5 + 32 + 64, 192), nn.SiLU(), nn.Linear(192, 192))
        self.to_pixels = nn.Conv2d(12, width, 3, padding=1)
        self.route = nn.Sequential(BottleneckAttention(32, heads=8), nn.Conv2d(32, width * 2, 1))
        self.route_gate = nn.Linear(192, width * 2)
        self.blocks = nn.ModuleList(RoutedBlock(width, d) for d in (1, 1, 2, 2, 4, 4, 2, 1))
        self.out = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, source, c64, q32, tid, strength, raw_stats, priors):
        amount = strength[:, None, None, None]
        aligned = apply_latent_chroma_shift(source, tid, priors, alpha=1.0)
        z0 = source + .35 * amount * (aligned - source)
        luma = torch.einsum("ij,bjhw->bihw", priors["P_Y"], source)
        style = self.stats(raw_stats[tid])
        cond = self.cond(torch.cat((F.one_hot(tid, 5).float(), sinusoidal_time_emb(strength, 32), style), 1))
        route16 = self.route(q32) * self.route_gate(cond).tanh()[:, :, None, None]
        route64 = F.interpolate(route16, size=(64, 64), mode="bilinear", align_corners=False)
        x = self.to_pixels(torch.cat((z0, luma, c64), 1).contiguous(memory_format=torch.channels_last))
        for block in self.blocks:
            x = block(x, cond, route64)
        return z0 + amount * self.out(F.silu(x))
