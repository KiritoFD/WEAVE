"""V36 CLEAVE-One Architectures:
1. StyleMLPEncoder: e_s = MLP_{1561->512->256}(F_style(x_r)) (<1M params, zero DINO).
2. FoundationPCoreNet: V34 C2-topology ~45.2M P-Core (base_dim=208, cond_dim=384) with
   randomly initialized weights, per-layer AdaGN conditioning, and optional activation checkpointing
   so RTX 3060 12GB stays <= 9.5GB without WDDM paging.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from exp.v29_biglittle.pcore_model import (
    BottleneckAttention,
    PureStyleResBlock,
    SPADEStyleBlock,
    sinusoidal_time_emb,
)


class StyleMLPEncoder(nn.Module):
    """Lightweight continuous style MLP: 1561 -> 512 -> 256 (<1M params)."""

    def __init__(
        self,
        in_dim: int = 1561,
        hidden_dim: int = 512,
        out_dim: int = 256,
        stat_mean: torch.Tensor | None = None,
        stat_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.register_buffer(
            "stat_mean",
            torch.zeros(1, in_dim) if stat_mean is None else stat_mean.float().view(1, in_dim),
        )
        self.register_buffer(
            "stat_std",
            torch.ones(1, in_dim) if stat_std is None else stat_std.float().clamp_min(1e-3).view(1, in_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def normalize_stats(self, stat: torch.Tensor) -> torch.Tensor:
        return ((stat.float() - self.stat_mean) / self.stat_std).clamp(-10.0, 10.0)

    def forward(self, stat: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.net(self.normalize_stats(stat)))


class FoundationPCoreNet(nn.Module):
    """V34 C2-class ~45.2M P-Core (base_dim=208, cond_dim=384) trained strictly from random init."""

    def __init__(
        self,
        base_dim: int = 208,
        style_dim: int = 256,
        cond_dim: int = 384,
        *,
        use_gamma: bool = True,
        normalize_style: bool = True,
        per_layer_adagn: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.style_dim = style_dim
        self.cond_dim = cond_dim
        self.use_gamma = use_gamma
        self.normalize_style = normalize_style
        self.per_layer_adagn = per_layer_adagn
        self.use_checkpoint = use_checkpoint

        w0, w1, w2 = base_dim, base_dim * 2, base_dim * 3
        dino_ch = 80

        self.dino_proj = nn.Sequential(
            nn.Conv2d(32, 160, 1),
            nn.GroupNorm(16, 160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, dino_ch, 1),
        )

        in_base = 64 + 2 + (16 if use_gamma else 0)
        self.base_cond_mlp = nn.Sequential(
            nn.Linear(in_base, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
        )
        self.style_cond_mlp = nn.Sequential(
            nn.Linear(style_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

        if per_layer_adagn:
            self.layer_style_heads = nn.ModuleDict({
                k: nn.Linear(style_dim, cond_dim)
                for k in ("enc64", "enc32", "mid1", "mid2", "dec32", "dec64")
            })
            for head in self.layer_style_heads.values():
                nn.init.normal_(head.weight, std=0.01)
                nn.init.zeros_(head.bias)
        else:
            self.layer_style_heads = None

        self.in_conv = nn.Conv2d(4, w0, 3, padding=1)
        self.enc64 = nn.ModuleList([PureStyleResBlock(w0, cond_dim), PureStyleResBlock(w0, cond_dim)])
        self.down32 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc32 = nn.ModuleList([PureStyleResBlock(w1, cond_dim), PureStyleResBlock(w1, cond_dim)])
        self.down16 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)

        s16_ch = 1 + dino_ch
        self.mid1 = SPADEStyleBlock(w2, s16_ch, cond_dim)
        self.mid_attn = BottleneckAttention(w2, heads=8)
        self.mid2 = SPADEStyleBlock(w2, s16_ch, cond_dim)

        self.up32 = nn.Conv2d(w2, w1, 3, padding=1)
        self.skip32 = nn.Conv2d(w1 * 2, w1, 1)
        self.dec32 = nn.ModuleList([PureStyleResBlock(w1, cond_dim), PureStyleResBlock(w1, cond_dim)])

        self.up64 = nn.Conv2d(w1, w0, 3, padding=1)
        self.skip64 = nn.Conv2d(w0 * 2, w0, 1)
        self.dec64 = nn.ModuleList([PureStyleResBlock(w0, cond_dim), PureStyleResBlock(w0, cond_dim)])

        self.out_norm = nn.GroupNorm(16, w0)
        self.out_conv = nn.Conv2d(w0, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _run_resblocks(self, blocks: nn.ModuleList, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            for blk in blocks:
                h = cp.checkpoint(blk, h, c, use_reentrant=False)
            return h
        for blk in blocks:
            h = blk(h, c)
        return h

    def _run_mid(self, h: torch.Tensor, s16: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            h = cp.checkpoint(self.mid1, h, s16, c1, use_reentrant=False)
            h = cp.checkpoint(self.mid_attn, h, use_reentrant=False)
            h = cp.checkpoint(self.mid2, h, s16, c2, use_reentrant=False)
            return h
        h = self.mid1(h, s16, c1)
        h = self.mid_attn(h)
        return self.mid2(h, s16, c2)

    def forward(
        self,
        z_tau: torch.Tensor,
        y16_lcn: torch.Tensor,
        q32_16: torch.Tensor,
        e_style: torch.Tensor,
        tau: torch.Tensor,
        *,
        drop_style: torch.Tensor | None = None,
        drop_content: torch.Tensor | None = None,
        layout_scale: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)

        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        q_in = q32_16
        y_in = y16_lcn
        if self.use_gamma:
            g_val = torch.ones(B, device=dev, dtype=torch.float32) if gamma is None else gamma.to(dev).float().view(B).clamp(0.0, 1.0)
            g4 = g_val.view(B, 1, 1, 1)
            q4 = F.avg_pool2d(q_in, 4).repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
            q4 = q4 - q4.mean(dim=(2, 3), keepdim=True)
            if self.training:
                q4 = q4 + (1.0 - g4) * 0.10 * torch.randn_like(q4)
            q_in = g4 * q_in + (1.0 - g4) * q4
            y_in = y_in * g4

        d_feat = self.dino_proj(q_in * cmask4) * lscale4
        s16 = torch.cat([y_in * lscale4, d_feat], dim=1)

        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        base_parts = [temb, flags]
        if self.use_gamma:
            base_parts.append(sinusoidal_time_emb(g_val, 16))
        c_base = self.base_cond_mlp(torch.cat(base_parts, dim=-1))

        s_vec = (F.normalize(e_style.float(), dim=-1) * 2.0 if self.normalize_style else e_style.float()) * smask2
        cond = c_base + self.style_cond_mlp(s_vec) * smask2

        def _c(name: str) -> torch.Tensor:
            if self.layer_style_heads is None:
                return cond
            return cond + self.layer_style_heads[name](s_vec) * smask2

        h64 = self.in_conv(z_tau)
        h64 = self._run_resblocks(self.enc64, h64, _c("enc64"))

        h32 = self.down32(h64)
        h32 = self._run_resblocks(self.enc32, h32, _c("enc32"))

        h16 = self.down16(h32)
        h16 = self._run_mid(h16, s16, _c("mid1"), _c("mid2"))

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        u32 = self._run_resblocks(self.dec32, u32, _c("dec32"))

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        u64 = self._run_resblocks(self.dec64, u64, _c("dec64"))

        return self.out_conv(F.silu(self.out_norm(u64)))
