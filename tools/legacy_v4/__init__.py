"""WEAVE-v4：真实 latent 条件扩散先验 + SDEdit 编辑。

与 v2/v3 完全独立的一条线：只学 ``p(z|s)``（真实作品自身的 latent 分布），
不做跨风格内容配对，不在训练侧接 VAE/DINO/CLIP。
"""
from __future__ import annotations

from .config import V4Config

__all__ = ["V4Config"]
