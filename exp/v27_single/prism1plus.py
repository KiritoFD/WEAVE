"""PRISM-1+ one-step editor. The content bottleneck is style-blind."""

import torch
import torch.nn.functional as F
from torch import nn

from exp.v18.ccf_model import BottleneckAttention


class SpatialAdaGN(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.norm = nn.GroupNorm(16, channels, affine=False)
        self.film = nn.Linear(cond_channels, channels * 2)
        self.spatial = nn.Conv2d(32, channels * 2, 1, bias=False)
        self.route = nn.Linear(cond_channels, channels * 2)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x, q, cond):
        routing = self.spatial(q) * self.route(cond).tanh()[:, :, None, None]
        routing = F.interpolate(routing, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gain, bias = (routing + self.film(cond)[:, :, None, None]).chunk(2, 1)
        return x + self.conv(F.silu(self.norm(x) * (1 + gain) + bias))


class Prism1Plus(nn.Module):
    def __init__(self, stats_dim):
        super().__init__()
        w0, w1, w2 = 96, 192, 288
        cond_dim = 192
        self.content = nn.Sequential(
            nn.Conv2d(33, w2, 3, padding=1), nn.GroupNorm(16, w2), nn.SiLU(),
            nn.Conv2d(w2, w2, 3, padding=1), BottleneckAttention(w2),
            nn.Conv2d(w2, w2, 3, padding=1), nn.SiLU(),
        )
        self.skip64 = nn.Sequential(nn.Conv2d(1, w0, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(w0, w0, 3, padding=1))
        self.skip32 = nn.Sequential(nn.Conv2d(w0, w1, 3, stride=2, padding=1), nn.SiLU(),
                                    nn.Conv2d(w1, w1, 3, padding=1))
        self.cond = nn.Sequential(nn.Linear(stats_dim + 6, cond_dim), nn.SiLU(),
                                  nn.Linear(cond_dim, cond_dim))
        self.up32 = nn.Conv2d(w2 + w1, w1, 3, padding=1)
        self.up64 = nn.Conv2d(w1 + w0 + 4, w0, 3, padding=1)
        self.dec32 = nn.ModuleList(SpatialAdaGN(w1, cond_dim) for _ in range(3))
        self.dec64 = nn.ModuleList(SpatialAdaGN(w0, cond_dim) for _ in range(3))
        self.out = nn.Sequential(nn.GroupNorm(16, w0), nn.SiLU(), nn.Conv2d(w0, 4, 3, padding=1))

    def forward(self, luma, y16, q32, tid, stats, strength, noise):
        gate = strength[:, None, None, None]
        skip64 = self.skip64(luma)
        skip32 = self.skip32(skip64)
        x = self.content(torch.cat((y16, q32), 1))
        cond = self.cond(torch.cat((F.one_hot(tid, 5).float(), stats, strength[:, None]), 1))
        x = self.up32(torch.cat((F.interpolate(x, scale_factor=2, mode="nearest"), (1-gate)*skip32), 1))
        for block in self.dec32:
            x = block(x, q32, cond)
        x = self.up64(torch.cat((F.interpolate(x, scale_factor=2, mode="nearest"),
                                 (1-gate)*skip64, gate*noise), 1))
        for block in self.dec64:
            x = block(x, q32, cond)
        return self.out(x)
