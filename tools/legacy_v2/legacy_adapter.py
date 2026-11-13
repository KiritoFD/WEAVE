"""V1 用：把旧 WEAVE 速度模型包成 **可求导** 的少步生成器。

旧 ``WEAVE.integrate_transport`` 带 ``@torch.no_grad()``，不能用于新目标。
这里用 2–4 步 Euler 直接调用速度模型；HH 保持输入不变（旧模型没有 HH 头）。
推理期 AdaIN 外挂已验证贡献≈0，故这里默认不启用，保证梯度只经 ODE 路径。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar


class DifferentiableLegacyGenerator(nn.Module):
    def __init__(self, velocity_model: nn.Module, *, num_steps: int = 2, keep_hh: bool = True) -> None:
        super().__init__()
        self.model = velocity_model
        self.num_steps = max(1, int(num_steps))
        self.keep_hh = bool(keep_hh)

    def forward(
        self,
        content: torch.Tensor,
        style_id,
        *,
        style_latent: torch.Tensor | None = None,
        style_stats: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        del style_stats
        x = content
        b = x.shape[0]
        dt = 1.0 / float(self.num_steps)
        for step in range(self.num_steps):
            t = torch.full((b,), (step + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.model(x, t=t, style_id=style_id, style_latent=style_latent)
            ll, lh, hl, hh = dwt2_haar(x.float())
            ll = ll + dt * v["ll"].float()
            lh = lh + dt * v["lh"].float()
            hl = hl + dt * v["hl"].float()
            if not self.keep_hh and "hh" in v:
                hh = hh + dt * v["hh"].float()
            x = idwt2_haar(ll, lh, hl, hh).to(dtype=content.dtype)
        return {
            "z_hat": x,
            "style_embed": torch.zeros(b, 1, device=x.device, dtype=x.dtype),
        }


__all__ = ["DifferentiableLegacyGenerator"]
