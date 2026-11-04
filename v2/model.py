"""WEAVE-v2 主干与单步生成器。

两种模式共用同一个 64→32→16→8 ResUNet：
- ``mode="generator"``：单步 ``z_hat = G(z_c, s)``，输出 ``content + ΔLL/ΔHF``；
  HH 不再关闭。
- ``mode="velocity"``：作为 V2（旧 FM + 新主干）的速度场，返回
  ``{"ll","lh","hl","hh"}`` 供旧 ``FlowMatchingObjective`` 直接消费。

风格条件：128-d style embedding + 逐层 AdaLN/FiLM；可选 8–16 个可学习
style token，只在 8×8 最低尺度做一次 cross-attention（不是 256-token Style
Memory 主通路）。AdaLN 门控初值 0，因此启用与否在初始化时严格等价。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wavelet import dwt2_haar, idwt2_haar

from .config import V2Config


def _ids(style_id: Any, batch: int, device: torch.device) -> torch.Tensor:
    if style_id is None:
        return torch.zeros(batch, dtype=torch.long, device=device)
    if torch.is_tensor(style_id):
        ids = style_id.to(device=device, dtype=torch.long).view(-1)
        if ids.numel() == 1 and batch > 1:
            ids = ids.expand(batch)
        return ids
    return torch.full((batch,), int(style_id), dtype=torch.long, device=device)


def _norm_groups(channels: int, requested: int) -> int:
    groups = max(1, min(int(requested), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class StyleConditionerV2(nn.Module):
    """style_id -> (128-d embedding, 可选 8–16 个 style token, 可选统计条件)。"""

    def __init__(self, cfg: V2Config, num_styles: int) -> None:
        super().__init__()
        self.num_styles = int(num_styles)
        self.embed_dim = int(cfg.style_embed_dim)
        self.num_tokens = int(cfg.style_tokens)
        self.embed = nn.Embedding(self.num_styles, self.embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        if self.num_tokens > 0:
            self.tokens = nn.Parameter(torch.randn(self.num_styles, self.num_tokens, self.embed_dim) * 0.02)
        else:
            self.tokens = None
        self.stats_dim = int(cfg.stats_dim)
        if self.stats_dim > 0:
            self.stats_proj = nn.Sequential(
                nn.Linear(self.stats_dim, self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            # 0 = 统计条件初始完全关闭；训练中可打开
            self.stats_gate = nn.Parameter(torch.tensor(float(cfg.style_stats_gate_init)))
        else:
            self.stats_proj = None
            self.stats_gate = None

    def forward(
        self,
        style_id: Any,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        style_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ids = _ids(style_id, batch, device).clamp(0, self.num_styles - 1)
        emb = self.embed(ids).to(dtype=dtype)
        if (
            self.stats_proj is not None
            and self.stats_gate is not None
            and torch.is_tensor(style_stats)
            and style_stats.numel() > 0
        ):
            bank = style_stats.to(device=device, dtype=dtype)
            vec = bank.index_select(0, ids)
            emb = emb + torch.tanh(self.stats_gate).to(dtype=dtype) * self.stats_proj(vec.float()).to(dtype=dtype)
        tokens = None
        if self.tokens is not None:
            tokens = self.tokens.index_select(0, ids).to(device=device, dtype=dtype)
        return emb, tokens


class FiLMResBlock(nn.Module):
    """conv → groupnorm/silu → FiLM(style) → conv → groupnorm/silu → 残差。"""

    def __init__(
        self,
        cin: int,
        cout: int,
        cond_dim: int,
        *,
        groups: int,
        adaln_enabled: bool = True,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_norm_groups(cin, groups), cin, affine=False)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(cout, groups), cout, affine=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.adaln_enabled = bool(adaln_enabled)
        if self.adaln_enabled:
            self.film = nn.Linear(cond_dim, cout * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.film = None
            self.gate = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).to(dtype=h.dtype).chunk(2, dim=1)
            g = torch.tanh(self.gate).to(dtype=h.dtype)
            h = h * (1.0 + g * scale[:, :, None, None]) + g * shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class LowestScaleAttention(nn.Module):
    """最低尺度 (8×8) 的全局 self-attention。"""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_norm_groups(dim, 8), dim, affine=False)
        self.heads = max(1, int(heads))
        while dim % self.heads != 0 and self.heads > 1:
            self.heads -= 1
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, c, hh, ww = h.shape
        x = self.norm(h).flatten(2).transpose(1, 2)
        q = k = v = x
        q = q.view(b, hh * ww, self.heads, c // self.heads).transpose(1, 2)
        k = k.view(b, hh * ww, self.heads, c // self.heads).transpose(1, 2)
        v = v.view(b, hh * ww, self.heads, c // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, hh * ww, c)
        y = self.out(y).transpose(1, 2).reshape(b, c, hh, ww)
        return h + y


class StyleTokenCrossAttention(nn.Module):
    """低尺度特征 query 可学习 style token；输出投影零初始化（初始 no-op）。"""

    def __init__(self, dim: int, num_tokens: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_norm_groups(dim, 8), dim, affine=False)
        self.heads = max(1, int(heads))
        while dim % self.heads != 0 and self.heads > 1:
            self.heads -= 1
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.num_tokens = int(num_tokens)

    def forward(self, h: torch.Tensor, tokens: torch.Tensor | None) -> torch.Tensor:
        if tokens is None or self.num_tokens <= 0:
            return h
        b, c, hh, ww = h.shape
        x = self.norm(h).flatten(2).transpose(1, 2)
        q = self.q(x).view(b, hh * ww, self.heads, c // self.heads).transpose(1, 2)
        k = self.k(tokens).view(b, self.num_tokens, self.heads, c // self.heads).transpose(1, 2)
        v = self.v(tokens).view(b, self.num_tokens, self.heads, c // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, hh * ww, c)
        y = self.out(y).transpose(1, 2).reshape(b, c, hh, ww)
        return h + y


class V2Backbone(nn.Module):
    """64→32→16→8 ResUNet；``mode`` 决定输出 residual 还是速度场。"""

    def __init__(
        self,
        cfg: V2Config,
        *,
        latent_channels: int = 4,
        num_styles: int = 5,
        time_dim: int = 256,
        mode: str = "generator",
    ) -> None:
        super().__init__()
        if mode not in {"generator", "velocity"}:
            raise ValueError(f"mode must be generator|velocity, got {mode!r}")
        self.cfg = cfg
        self.mode = mode
        self.latent_channels = int(latent_channels)
        self.num_styles = int(num_styles)
        self.time_dim = int(time_dim)
        channels = list(cfg.channels)
        self.channels = channels
        cond_dim = int(cfg.style_embed_dim) + (self.time_dim if mode == "velocity" else 0)
        self.style = StyleConditionerV2(cfg, self.num_styles)

        self.stem = nn.Conv2d(self.latent_channels, channels[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for level in range(4):
            blocks = nn.ModuleList(
                [
                    FiLMResBlock(
                        channels[level],
                        channels[level],
                        cond_dim,
                        groups=cfg.norm_groups,
                        adaln_enabled=cfg.adaln_enabled,
                        gate_init=cfg.adaln_gate_init,
                    )
                    for _ in range(int(cfg.blocks_per_stage[level]))
                ]
            )
            self.enc_blocks.append(blocks)
            if level < 3:
                self.downs.append(nn.Conv2d(channels[level], channels[level + 1], 3, stride=2, padding=1))
        self.lowest_attn = (
            LowestScaleAttention(channels[3], cfg.attn_heads) if cfg.attn_heads > 0 else nn.Identity()
        )
        if cfg.style_tokens > 0:
            self.token_proj = nn.Linear(int(cfg.style_embed_dim), channels[3])
            self.style_cross = StyleTokenCrossAttention(channels[3], cfg.style_tokens, cfg.style_token_heads)
        else:
            self.token_proj = None
            self.style_cross = None

        self.up_convs = nn.ModuleList([nn.Conv2d(channels[i + 1], channels[i], 3, padding=1) for i in (2, 1, 0)])
        self.dec_blocks = nn.ModuleList()
        for level in (2, 1, 0):
            blocks = nn.ModuleList(
                [
                    FiLMResBlock(
                        channels[level] * 2,
                        channels[level],
                        cond_dim,
                        groups=cfg.norm_groups,
                        adaln_enabled=cfg.adaln_enabled,
                        gate_init=cfg.adaln_gate_init,
                    )
                    for _ in range(int(cfg.decoder_blocks_per_stage[level]))
                ]
            )
            self.dec_blocks.append(blocks)

        out_ch = 4 if mode == "generator" else 16
        if mode == "generator":
            self.head_ll = nn.Conv2d(channels[0], 4, 3, padding=1)
            self.head_hf = nn.Conv2d(channels[0], 12, 3, padding=1)
            for head in (self.head_ll, self.head_hf):
                if bool(getattr(cfg, "residual_head_zero_init", False)):
                    # V3-SWD：零初始化输出头，初始映射严格 = 恒等
                    nn.init.zeros_(head.weight)
                else:
                    nn.init.normal_(head.weight, std=float(cfg.residual_init_std))
                nn.init.zeros_(head.bias)
        else:
            self.head_ll = nn.Conv2d(channels[0], 4, 3, padding=1)
            self.head_lh = nn.Conv2d(channels[0], 4, 3, padding=1)
            self.head_hl = nn.Conv2d(channels[0], 4, 3, padding=1)
            self.head_hh = nn.Conv2d(channels[0], 4, 3, padding=1)
            for head in (self.head_ll, self.head_lh, self.head_hl, self.head_hh):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        self.out_channels = out_ch
        self.time_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(self.time_dim, self.time_dim))
            if mode == "velocity"
            else None
        )

    def _time_embed(self, t: Any, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t is None:
            t = torch.zeros(batch, device=device, dtype=dtype)
        elif isinstance(t, (int, float)):
            t = torch.full((batch,), float(t), device=device, dtype=dtype)
        else:
            t = t.to(device=device, dtype=dtype).view(-1)
            if t.numel() == 1 and batch > 1:
                t = t.expand(batch)
        half = self.time_dim // 2
        freq = torch.arange(half, device=device, dtype=torch.float32)
        freq = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * freq / max(1, half))
        ang = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if emb.shape[1] < self.time_dim:
            emb = F.pad(emb, (0, self.time_dim - emb.shape[1]))
        return self.time_proj(emb).to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: Any = None,
        style_id: Any = None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        cfg_unconditional: bool = False,
        reference_feats: torch.Tensor | None = None,
        style_stats: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        del style_latent, style_text_tokens, cfg_unconditional, reference_feats
        if x.dim() != 4:
            raise ValueError(f"v2 backbone expects (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.latent_channels:
            raise ValueError(f"expected {self.latent_channels} latent channels, got {x.shape[1]}")
        b = x.shape[0]
        style_emb, style_tokens = self.style(
            style_id, batch=b, device=x.device, dtype=x.dtype, style_stats=style_stats
        )
        if self.mode == "velocity":
            cond = torch.cat([style_emb, self._time_embed(t, b, x.device, x.dtype)], dim=1)
        else:
            cond = style_emb

        h = self.stem(x)
        feats = [h]
        for level in range(3):
            for blk in self.enc_blocks[level]:
                h = blk(h, cond)
            feats.append(h)
            h = self.downs[level](h)
        for blk in self.enc_blocks[3]:
            h = blk(h, cond)
        h = self.lowest_attn(h)
        if self.style_cross is not None:
            if style_tokens is not None and self.token_proj is not None:
                style_tokens = self.token_proj(style_tokens)
            h = self.style_cross(h, style_tokens)

        for i, level in enumerate((2, 1, 0)):
            h = self.up_convs[i](F.interpolate(h, scale_factor=2, mode="nearest"))
            # feats[0] 是 stem；每个 level 的输出在 feats[level+1]
            h = torch.cat([h, feats[level + 1]], dim=1)
            for blk in self.dec_blocks[i]:
                h = blk(h, cond)

        if self.mode == "velocity":
            h_sub = F.avg_pool2d(h, 2)
            return {
                "ll": self.head_ll(h_sub),
                "lh": self.head_lh(h_sub),
                "hl": self.head_hl(h_sub),
                "hh": self.head_hh(h_sub),
            }

        content = x.float()
        ll, lh, hl, hh = dwt2_haar(content)
        # 残差在子带分辨率 (H/2, W/2) 上预测，与 LL/LH/HL/HH 对齐
        h_sub = F.avg_pool2d(h, 2) if h.shape[-2] == content.shape[-2] else h
        d_ll = self.head_ll(h_sub) * float(self.cfg.residual_ll_scale)
        d_hf = self.head_hf(h_sub) * float(self.cfg.residual_hf_scale)
        d_lh, d_hl, d_hh = d_hf.chunk(3, dim=1)
        z_hat = idwt2_haar(ll + d_ll, lh + d_lh, hl + d_hl, hh + d_hh)
        return {
            "z_hat": z_hat.to(dtype=x.dtype),
            "delta_ll": d_ll.to(dtype=x.dtype),
            "delta_lh": d_lh.to(dtype=x.dtype),
            "delta_hl": d_hl.to(dtype=x.dtype),
            "delta_hh": d_hh.to(dtype=x.dtype),
            "style_embed": style_emb,
        }


class V2Generator(nn.Module):
    """显式单步生成器包装，接口固定为 ``G(content, style_id, style_stats)``。"""

    def __init__(self, cfg: V2Config, *, latent_channels: int = 4, num_styles: int = 5) -> None:
        super().__init__()
        self.backbone = V2Backbone(
            cfg, latent_channels=latent_channels, num_styles=num_styles, mode="generator"
        )
        self.cfg = cfg

    def forward(
        self,
        content: torch.Tensor,
        style_id: Any,
        *,
        style_stats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.backbone(content, style_id=style_id, style_stats=style_stats)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "StyleConditionerV2",
    "FiLMResBlock",
    "LowestScaleAttention",
    "StyleTokenCrossAttention",
    "V2Backbone",
    "V2Generator",
    "count_parameters",
]
