"""V8 conditional probability-flow velocity U-Net.

No ScoreNet / DMD. The model predicts u_theta(z_tau, tau, s, t).
Same-style exact identity is handled at the solver interface, not by a loss.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.v6.models_v6 import StyleAdaLNBlock


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
    args = t.float().view(-1, 1) * freqs.view(1, -1) * 2.0 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


class VelocityUNet(nn.Module):
    def __init__(self, base_dim: int = 48, num_styles: int = 5):
        super().__init__()
        c1, c2, c3 = base_dim, base_dim * 2, base_dim * 2
        cond_dim = base_dim * 2
        self.base_dim = base_dim
        self.style_emb = nn.Embedding(num_styles, cond_dim)
        self.pair_mlp = nn.Sequential(nn.Linear(cond_dim * 3, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.time_mlp = nn.Sequential(nn.Linear(base_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        self.stem = nn.Conv2d(4, c1, 3, padding=1)
        self.enc1_1 = StyleAdaLNBlock(c1, cond_dim, content_channels=0)
        self.enc1_2 = StyleAdaLNBlock(c1, cond_dim, content_channels=0)
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2_1 = StyleAdaLNBlock(c2, cond_dim, content_channels=0)
        self.enc2_2 = StyleAdaLNBlock(c2, cond_dim, content_channels=0)
        self.down4 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid_1 = StyleAdaLNBlock(c3, cond_dim, content_channels=0)
        self.mid_2 = StyleAdaLNBlock(c3, cond_dim, content_channels=0)
        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.dec2_proj = nn.Conv2d(c2 * 2, c2, 1)
        self.dec2_1 = StyleAdaLNBlock(c2, cond_dim, content_channels=0)
        self.dec2_2 = StyleAdaLNBlock(c2, cond_dim, content_channels=0)
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.dec1_proj = nn.Conv2d(c1 * 2, c1, 1)
        self.dec1_1 = StyleAdaLNBlock(c1, cond_dim, content_channels=0)
        self.dec1_2 = StyleAdaLNBlock(c1, cond_dim, content_channels=0)
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_tau: torch.Tensor, tau: torch.Tensor, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        B = z_tau.shape[0]
        if not torch.is_tensor(tau):
            tau = torch.full((B,), float(tau), device=z_tau.device, dtype=torch.float32)
        s = self.style_emb(src_ids.long())
        t = self.style_emb(tgt_ids.long())
        cond = self.pair_mlp(torch.cat([t - s, s, t], dim=-1)) + self.time_mlp(
            sinusoidal_embedding(tau.float().view(B), self.base_dim)
        )
        h1 = self.enc1_2(self.enc1_1(self.stem(z_tau), cond), cond)
        h2 = self.down2(h1)
        h2 = self.enc2_2(self.enc2_1(h2, cond), cond)
        h4 = self.down4(h2)
        h4 = self.mid_2(self.mid_1(h4, cond), cond)
        u2 = self.up2(F.interpolate(h4, scale_factor=2.0, mode="nearest"))
        u2 = self.dec2_proj(torch.cat([u2, h2], dim=1))
        u2 = self.dec2_2(self.dec2_1(u2, cond), cond)
        u1 = self.up1(F.interpolate(u2, scale_factor=2.0, mode="nearest"))
        u1 = self.dec1_proj(torch.cat([u1, h1], dim=1))
        u1 = self.dec1_2(self.dec1_1(u1, cond), cond)
        return self.out_conv(F.silu(self.out_norm(u1)))
