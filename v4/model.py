"""WEAVE-v4 主干：4 尺度 ResUNet，输出 4 通道噪声预测。

条件 = 正弦时间嵌入 + Style-ID embedding，注入方式为逐层 FiLM（scale/shift）。
FiLM 输出层零初始化：初始恒等，但梯度非零（scale 的梯度 ∝ 归一化特征）。
Attention 只在 16×16 与 8×8；不引入 Style Memory。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import V4Config


def timestep_embedding(t: torch.Tensor, dim: int, *, max_period: int = 10000) -> torch.Tensor:
    """正弦嵌入；``t`` 为 [B] float。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FiLM(nn.Module):
    """条件 -> (scale, shift)；默认零初始化输出层（初始恒等），可配置非零初始化。"""

    def __init__(self, cond_dim: int, channels: int, init_std: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        if init_std > 0.0:
            nn.init.normal_(self.proj.weight, std=init_std)
        else:
            nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, *, norm_groups: int = 8, dropout: float = 0.0,
                 film_init_std: float = 0.0, style_cond_dim: int = 0,
                 style_film_init_std: float = 0.05) -> None:
        super().__init__()
        groups = norm_groups if channels % norm_groups == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film1 = FiLM(cond_dim, channels, film_init_std)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film2 = FiLM(cond_dim, channels, film_init_std)
        # split 注入：时间与风格解耦，风格走独立逐层 FiLM（E2）
        self.style_film1 = FiLM(style_cond_dim, channels, style_film_init_std) if style_cond_dim else None
        self.style_film2 = FiLM(style_cond_dim, channels, style_film_init_std) if style_cond_dim else None
        self.dropout = dropout

    def forward(self, h: torch.Tensor, cond: torch.Tensor,
                style_cond: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv1(F.silu(self.norm1(h)))
        x = self.film1(x, cond)
        if self.style_film1 is not None and style_cond is not None:
            x = self.style_film1(x, style_cond)
        x = self.conv2(F.silu(self.norm2(x)))
        x = self.film2(x, cond)
        if self.style_film2 is not None and style_cond is not None:
            x = self.style_film2(x, style_cond)
        if self.dropout > 0.0:
            x = F.dropout(x, self.dropout, self.training)
        return h + x


class SelfAttention2d(nn.Module):
    """SAGAN 风格空间自注意力，残余缩放 1/sqrt(head_dim)。"""

    def __init__(self, channels: int, *, heads: int = 4, norm_groups: int = 8) -> None:
        super().__init__()
        groups = norm_groups if channels % norm_groups == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.heads = heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.scale = (channels // heads) ** -0.5

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, c, hh, ww = h.shape
        q, k, v = self.qkv(self.norm(h)).reshape(b, 3, self.heads, c // self.heads, hh * ww).unbind(1)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = (attn @ v).reshape(b, c, hh, ww)
        return h + self.proj(out)


def _make_resblock(channels: int, cfg: V4Config, cond_dim: int, ng: int) -> ResBlock:
    """按 cond_mode 构造 ResBlock：split 时每块额外带风格 FiLM。"""
    split = cfg.cond_mode == "split"
    return ResBlock(
        channels, cond_dim, norm_groups=ng, dropout=cfg.dropout,
        film_init_std=cfg.film_init_std,
        style_cond_dim=cfg.style_film_dim if split else 0,
        style_film_init_std=cfg.style_film_init_std,
    )


class V4UNet(nn.Module):
    """64×64 latent -> 噪声预测；4 尺度，FiLM 条件。"""

    def __init__(self, cfg: V4Config) -> None:
        super().__init__()
        ch = cfg.channels
        c0, c1, c2, c3 = ch
        cond_dim = cfg.cond_dim
        ng = cfg.norm_groups
        self.cond_mode = cfg.cond_mode

        self.time_embed = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_dim = cfg.time_embed_dim
        self.style_embed = nn.Embedding(cfg.num_styles, cfg.style_embed_dim)
        self.null_style = nn.Parameter(torch.zeros(cfg.style_embed_dim))
        self.style_proj = nn.Sequential(
            nn.Linear(cfg.style_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        nn.init.normal_(self.style_embed.weight, std=0.02)
        self.style_film_proj = None
        if cfg.cond_mode == "split":
            self.style_film_proj = nn.Sequential(
                nn.Linear(cfg.style_embed_dim, cfg.style_film_dim),
                nn.SiLU(),
                nn.Linear(cfg.style_film_dim, cfg.style_film_dim),
            )
            nn.init.normal_(self.style_film_proj[0].weight, std=0.02)
            nn.init.normal_(self.style_film_proj[2].weight, std=0.02)

        self.input_conv = nn.Conv2d(cfg.latent_channels, c0, 3, padding=1)
        self.stage0 = nn.ModuleList([_make_resblock(c0, cfg, cond_dim, ng)
                                     for _ in range(cfg.blocks_per_stage[0])])
        self.down0 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.stage1 = nn.ModuleList([_make_resblock(c1, cfg, cond_dim, ng)
                                     for _ in range(cfg.blocks_per_stage[1])])
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.stage2 = nn.ModuleList([_make_resblock(c2, cfg, cond_dim, ng)
                                     for _ in range(cfg.blocks_per_stage[2])])
        self.attn2 = SelfAttention2d(c2, heads=cfg.attn_heads, norm_groups=ng)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.stage3 = nn.ModuleList([_make_resblock(c3, cfg, cond_dim, ng)
                                     for _ in range(cfg.blocks_per_stage[3])])
        self.attn3 = SelfAttention2d(c3, heads=cfg.attn_heads, norm_groups=ng)

        self.mid = nn.ModuleList([_make_resblock(c3, cfg, cond_dim, ng)
                                  for _ in range(cfg.mid_blocks)])
        self.mid_attn = SelfAttention2d(c3, heads=cfg.attn_heads, norm_groups=ng)

        self.up3 = nn.Conv2d(c3, c2, 3, padding=1)
        self.up3_res = nn.ModuleList([_make_resblock(c2, cfg, cond_dim, ng)])
        self.up3_attn = SelfAttention2d(c2, heads=cfg.attn_heads, norm_groups=ng)
        self.up2 = nn.Conv2d(c2 + c1, c1, 3, padding=1)
        self.up2_res = nn.ModuleList([_make_resblock(c1, cfg, cond_dim, ng)])
        self.up1 = nn.Conv2d(c1 + c0, c0, 3, padding=1)
        self.up1_res = nn.ModuleList([_make_resblock(c0, cfg, cond_dim, ng)])

        self.out_norm = nn.GroupNorm(ng if c0 % ng == 0 else 1, c0)
        self.out_conv = nn.Conv2d(c0, cfg.latent_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _cond_parts(self, t: torch.Tensor,
                    style_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        temb = self.time_embed(timestep_embedding(t, self.time_dim).to(t.dtype))
        sid = style_id.clamp(min=0)
        semb = self.style_embed(sid)
        null = self.null_style.to(semb.dtype).unsqueeze(0).expand_as(semb)
        semb = torch.where((style_id < 0).unsqueeze(-1), null, semb)
        style = self.style_proj(semb)
        if self.cond_mode == "split" and self.style_film_proj is not None:
            return temb, self.style_film_proj(semb)
        return temb + style, None

    def _cond(self, t: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        """诊断用：返回主干共用的条件向量（split 模式下退化为时间条件）。"""
        cond, _ = self._cond_parts(t, style_id)
        return cond

    def forward(self, x: torch.Tensor, t: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        cond, style_cond = self._cond_parts(t, style_id)
        h0 = self.input_conv(x)
        for blk in self.stage0:
            h0 = blk(h0, cond, style_cond)
        h1 = self.down0(h0)
        for blk in self.stage1:
            h1 = blk(h1, cond, style_cond)
        h2 = self.down1(h1)
        for blk in self.stage2:
            h2 = blk(h2, cond, style_cond)
        h2 = self.attn2(h2)
        h3 = self.down2(h2)
        for blk in self.stage3:
            h3 = blk(h3, cond, style_cond)
        h3 = self.attn3(h3)

        mid = h3
        for blk in self.mid:
            mid = blk(mid, cond, style_cond)
        mid = self.mid_attn(mid)

        up3 = F.interpolate(mid, scale_factor=2, mode="nearest")
        up3 = self.up3(up3) + h2
        for blk in self.up3_res:
            up3 = blk(up3, cond, style_cond)
        up3 = self.up3_attn(up3)
        up2 = F.interpolate(up3, scale_factor=2, mode="nearest")
        up2 = self.up2(torch.cat([up2, h1], dim=1))
        for blk in self.up2_res:
            up2 = blk(up2, cond, style_cond)
        up1 = F.interpolate(up2, scale_factor=2, mode="nearest")
        up1 = self.up1(torch.cat([up1, h0], dim=1))
        for blk in self.up1_res:
            up1 = blk(up1, cond, style_cond)
        return self.out_conv(F.silu(self.out_norm(up1)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["V4UNet", "timestep_embedding", "count_parameters", "FiLM", "ResBlock", "SelfAttention2d"]

