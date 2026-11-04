"""条件多尺度 PatchGAN 判别器。

- 有限感受野：3 层 conv4x4(stride 2/1) 的 Markovian patch 判别器；
- 多尺度：同一份共享权重作用在 1×/2×/4× 下采样输入上；
- 条件：style_id -> 128-d embedding，经 FiLM 注入每个 conv 块。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import V2Config
from .model import _ids, _norm_groups


class _CondPatchBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim: int, *, stride: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 4, stride=stride, padding=1)
        self.norm = nn.GroupNorm(_norm_groups(cout, groups), cout, affine=False)
        self.film = nn.Linear(cond_dim, cout * 2)
        nn.init.normal_(self.film.weight, std=0.02)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        scale, shift = self.film(cond).to(dtype=h.dtype).chunk(2, dim=1)
        h = self.norm(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return F.silu(h)


class ConditionalPatchDiscriminator(nn.Module):
    def __init__(
        self,
        cfg: V2Config,
        *,
        latent_channels: int = 4,
        num_styles: int = 5,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_styles = int(num_styles)
        cond_dim = int(cfg.disc_style_dim)
        self.style_embed = nn.Embedding(self.num_styles, cond_dim)
        nn.init.normal_(self.style_embed.weight, std=0.02)
        base = int(cfg.disc_base_channels)
        layers = [base, base * 2, base * 4]
        self.blocks = nn.ModuleList()
        cin = latent_channels
        for idx, cout in enumerate(layers[: max(1, int(cfg.disc_layers))]):
            stride = 2 if idx < 2 else 1
            self.blocks.append(
                _CondPatchBlock(cin, cout, cond_dim, stride=stride, groups=cfg.norm_groups)
            )
            cin = cout
        self.out = nn.Conv2d(cin, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, style_id: Any) -> list[torch.Tensor]:
        b = x.shape[0]
        ids = _ids(style_id, b, x.device).clamp(0, self.num_styles - 1)
        cond = self.style_embed(ids).to(dtype=x.dtype)
        logits: list[torch.Tensor] = []
        for scale in range(int(self.cfg.disc_num_scales)):
            if scale == 0:
                xi = x
            else:
                xi = F.avg_pool2d(x, kernel_size=2 ** scale, ceil_mode=False)
                if xi.shape[-2] < 8 or xi.shape[-1] < 8:
                    break
            h = xi
            for blk in self.blocks:
                h = blk(h, cond)
            logits.append(self.out(h))
        return logits

    def score(self, x: torch.Tensor, style_id: Any) -> torch.Tensor:
        """所有尺度的 patch logits 均值，便于 hinge/LSGAN 损失。"""
        parts = self.forward(x, style_id)
        return torch.stack([p.float().mean() for p in parts]).mean()


def adv_g_loss(d_logits_fake: torch.Tensor, mode: str = "hinge") -> torch.Tensor:
    if mode == "lsgan":
        return 0.5 * ((d_logits_fake - 1.0) ** 2).mean()
    return -d_logits_fake.mean()


def adv_d_loss(
    d_logits_real: torch.Tensor,
    d_logits_fake: torch.Tensor,
    mode: str = "hinge",
) -> torch.Tensor:
    if mode == "lsgan":
        return 0.5 * ((d_logits_real - 1.0) ** 2).mean() + 0.5 * (d_logits_fake ** 2).mean()
    return F.relu(1.0 - d_logits_real).mean() + F.relu(1.0 + d_logits_fake).mean()


__all__ = [
    "ConditionalPatchDiscriminator",
    "adv_g_loss",
    "adv_d_loss",
]
