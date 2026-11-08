"""WEAVE V7 model: single-step conditional residual distribution transport operator.

T_theta(z,s,t) = z + 1_{s != t} * R_theta(z, e_t - e_s)

Unlike V6 D1, the network sees the ORIGINAL latent z (not only C1 canonical content);
C1/edge canonicalization is used only as a soft transport cost during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.v6.models_v6 import StyleAdaLNBlock


class SingleStepTransportNet(nn.Module):
    """Lightweight pair-conditioned U-Net residual operator (~2M params at base_dim=48)."""

    def __init__(self, base_dim: int = 48, num_styles: int = 5):
        super().__init__()
        c1, c2, c3 = base_dim, base_dim * 2, base_dim * 2
        cond_dim = base_dim * 2

        self.style_emb = nn.Embedding(num_styles, cond_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(cond_dim * 3, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

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

    def forward(
        self,
        z: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        s_emb = self.style_emb(src_ids.long())
        t_emb = self.style_emb(tgt_ids.long())
        cond = self.pair_mlp(torch.cat([t_emb - s_emb, s_emb, t_emb], dim=-1))

        h1 = self.enc1_2(self.enc1_1(self.stem(z), cond), cond)
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

        residual = self.out_conv(F.silu(self.out_norm(u1)))
        same_mask = (src_ids == tgt_ids).view(-1, 1, 1, 1)
        # Exact same-style identity by construction.
        return z + torch.where(same_mask, torch.zeros_like(residual), residual)
