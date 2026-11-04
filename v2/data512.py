"""512 packed latent 池（4x64x64）读取；内容图与风格图都来自同一份训练集。

不做任何池化/重编码：直接读 ``data/train/.latent_cache/packed/packed/*.pt``。
每个样本返回 content latent 与它的源风格 id，供 Identity 项与内容监督使用。
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


def _style_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no packed latent files under {root}")
    return files


class StyleLatentPool:
    """五类 latent 常驻内存；``stack[style_id]`` 是 [N,4,64,64] float32。"""

    def __init__(self, root: str | Path, *, dtype: torch.dtype = torch.float32) -> None:
        self.root = Path(root)
        self.files = _style_files(self.root)
        stacks: list[torch.Tensor] = []
        self.style_names: list[str] = []
        for path in self.files:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            latents = payload["latents"] if isinstance(payload, dict) else payload
            if latents.dim() != 4 or latents.shape[1:] != (4, 64, 64):
                raise ValueError(f"{path.name}: expect [N,4,64,64], got {tuple(latents.shape)}")
            stacks.append(latents.to(dtype).contiguous())
            self.style_names.append(path.stem.split("_", 1)[-1] if "_" in path.stem else path.stem)
        if len(stacks) != 5:
            raise ValueError(f"expect 5 style packs, got {len(stacks)}: {[p.name for p in self.files]}")
        self.stack = stacks
        self.num_styles = len(stacks)
        self.total = sum(s.shape[0] for s in stacks)
        self._all = torch.cat(stacks, dim=0)
        self._all_style = torch.cat(
            [torch.full((s.shape[0],), i, dtype=torch.long) for i, s in enumerate(stacks)]
        )

    def sample_content(self, batch: int, *, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(0, self.total, (batch,), generator=gen)
        return self._all.index_select(0, idx).clone()

    def sample_style(self, style_id: int, batch: int, *, gen: torch.Generator) -> torch.Tensor:
        pool = self.stack[int(style_id)]
        idx = torch.randint(0, pool.shape[0], (batch,), generator=gen)
        return pool.index_select(0, idx).clone()


class LatentPoolDataset(Dataset):
    """给 DataLoader 用的轻量包装（返回 content + source_style_id）。"""

    def __init__(self, pool: StyleLatentPool) -> None:
        self.pool = pool

    def __len__(self) -> int:
        return self.pool.total

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"content": self.pool._all[idx], "source_style_id": self.pool._all_style[idx]}


__all__ = ["StyleLatentPool", "LatentPoolDataset"]
