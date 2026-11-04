"""轻量 Style Encoder（只在真实作品上训练，之后冻结）。

用途：手写 Φ（颜色/局部对比度/方向纹理/笔触）在五类上类内/类间可分性不足时，
补一个 128-d 风格嵌入（Style-ID 分类 + 同风格监督对比），全程不参与对抗训练。

网络：128px 输入 → 4 个 stride-2 卷积块（32/64/128/128）→ GAP → 128-d → L2 归一化。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class StyleEncoder(nn.Module):
    def __init__(self, num_styles: int = 5, embed_dim: int = 128) -> None:
        super().__init__()
        c = (32, 64, 128, 128)
        self.stem = nn.Conv2d(3, c[0], 3, stride=2, padding=1)
        self.blocks = nn.ModuleList()
        for i in range(3):
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(min(8, c[i]), c[i]),
                    nn.SiLU(),
                    nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1),
                    nn.GroupNorm(min(8, c[i + 1]), c[i + 1]),
                    nn.SiLU(),
                    nn.Conv2d(c[i + 1], c[i + 1], 3, padding=1),
                )
            )
        self.proj = nn.Linear(c[-1], embed_dim)
        self.head = nn.Linear(embed_dim, num_styles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (L2 归一化嵌入用于对比, 未归一化 logits 用于分类)。

        分类头必须建在未归一化特征上：归一化会把梯度压到 1/||h||，实测会导致
        “嵌入塌成常数 + CE 卡在 ln(5)” 的伪收敛。
        """
        h = F.silu(self.stem(x))
        for blk in self.blocks:
            h = blk(h)
        h = h.mean(dim=(2, 3))
        z = self.proj(h)
        return F.normalize(z, dim=1), self.head(z)


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """同风格为正样本的监督对比损失（NT-Xent 形式）。"""
    sim = emb @ emb.t() / temperature
    n = sim.shape[0]
    mask = labels[:, None] == labels[None, :]
    mask.fill_diagonal_(False)
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(logits)
    log_prob = logits - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)
    pos = exp * mask
    denom = pos.sum(dim=1)
    valid = denom > 0
    if not bool(valid.any()):
        return emb.new_zeros(())
    return -(log_prob * mask).sum(dim=1)[valid].div(denom[valid]).mean()


__all__ = ["StyleEncoder", "supcon_loss"]
