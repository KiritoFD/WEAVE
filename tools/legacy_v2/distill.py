"""可选第二阶段：用训练好的单步生成器 G_φ 作为 stopgrad 教师构造端点，
蒸馏回一个 Flow-Matching 速度模型。

默认关闭；只有显式 ``v2.distill_enabled=true`` 且提供教师 checkpoint 时才运行。
这里不假设蒸馏一定优于直接单步生成。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tools.legacy_root.wavelet import dwt2_haar

from .config import V2Config


class DistillObjective(nn.Module):
    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.cfg = cfg

    def make_endpoint(
        self,
        teacher: nn.Module,
        content: torch.Tensor,
        style_id: Any,
        *,
        style_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            out = teacher(content, style_id=style_id, style_stats=style_stats)
        return out["z_hat"].detach()

    def compute(
        self,
        student: nn.Module,
        teacher: nn.Module,
        content: torch.Tensor,
        style_id: Any,
        *,
        style_stats: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        z_end = self.make_endpoint(teacher, content, style_id, style_stats=style_stats)
        b = content.shape[0]
        t = torch.rand(b, device=content.device, generator=generator)
        t_b = t.view(b, 1, 1, 1)
        x_t = (1.0 - t_b) * content + t_b * z_end
        v_gt = z_end - content

        pred = student(x_t, t=t, style_id=style_id)
        gt_ll, gt_lh, gt_hl, gt_hh = dwt2_haar(v_gt.float())
        loss = (
            F.mse_loss(pred["ll"].float(), gt_ll)
            + F.mse_loss(pred["lh"].float(), gt_lh)
            + F.mse_loss(pred["hl"].float(), gt_hl)
            + F.mse_loss(pred["hh"].float(), gt_hh)
        )
        return {"loss": float(self.cfg.distill_weight) * loss, "loss_distill": loss.detach()}


__all__ = ["DistillObjective"]
