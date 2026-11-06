"""伪GT patch 掩码构造（gt-practice §1.3）。

九种配置 P1–P9 全部在 512 图像坐标下定义，坐标对齐 16px 网格（因为 Haar 一级子带
分辨率是 32×32，图像/16 = 子带坐标）。同一配置内的 patch 互不重叠。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

IMAGE_SIZE = 512
GRID = 16
LATENT_SIZE = 64
SUBBAND_SIZE = 32

#: 名义配置：name -> (patch 边长列表, 目标覆盖率)
PATCH_SPECS: Dict[str, Dict[str, object]] = {
    "P1": {"sizes": [64] * 8, "coverage": 0.125},
    "P2": {"sizes": [64] * 16, "coverage": 0.25},
    "P3": {"sizes": [64] * 32, "coverage": 0.50},
    "P4": {"sizes": [128] * 2, "coverage": 0.125},
    "P5": {"sizes": [128] * 4, "coverage": 0.25},
    "P6": {"sizes": [128] * 8, "coverage": 0.50},
    "P7": {"sizes": [256] * 1, "coverage": 0.25},
    "P8": {"sizes": [256] * 2, "coverage": 0.50},
    "P9": {"sizes": [128] + [64] * 12, "coverage": 0.25},
}

#: 反向污染 E 的 7 种配置（复用上面的几何，Feather/Hard 由调用方决定）
E_SPECS: Dict[str, Dict[str, object]] = {
    "E1": {"sizes": [64] * 16, "coverage": 0.25, "feather": True},
    "E2": {"sizes": [64] * 32, "coverage": 0.50, "feather": True},
    "E3": {"sizes": [128] * 4, "coverage": 0.25, "feather": True},
    "E4": {"sizes": [128] * 8, "coverage": 0.50, "feather": True},
    "E5": {"sizes": [256] * 1, "coverage": 0.25, "feather": True},
    "E6": {"sizes": [128] + [64] * 12, "coverage": 0.25, "feather": True},
    "E7": {"sizes": [128] * 4, "coverage": 0.25, "feather": False},
}


@dataclass(frozen=True)
class Rect:
    """图像坐标下的矩形。x/y 为左上角，w/h 为尺寸（像素）。"""

    x: int
    y: int
    w: int
    h: int

    @property
    def tokens(self) -> tuple[int, int, int, int]:
        """DINO/Haar 网格坐标 (x, y, w, h)，按 16px 网格对齐。"""
        return (self.x // GRID, self.y // GRID, self.w // GRID, self.h // GRID)

    @property
    def area(self) -> int:
        return int(self.w) * int(self.h)


def _overlaps(a: Rect, b: Rect) -> bool:
    return not (a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y)


def build_patch_layout(
    spec: str,
    *,
    image_size: int = IMAGE_SIZE,
    grid: int = GRID,
    seed: int = 0,
    max_tries: int = 4000,
) -> List[Rect]:
    """按配置名生成互不重叠、16px 对齐的 patch 列表。

    放置顺序 = 先大后小（256 → 128 → 64），保证混合尺度 P9 能放进去。
    候选位置先按 ``p`` 步长铺满并打乱（保证 P8 这类高覆盖率配置一定能放下），
    再退回随机对齐位置。同一 ``seed`` 逐位可重现。
    """
    if spec in PATCH_SPECS:
        sizes = list(PATCH_SPECS[spec]["sizes"])  # type: ignore[arg-type]
    elif spec in E_SPECS:
        sizes = list(E_SPECS[spec]["sizes"])  # type: ignore[arg-type]
    else:
        raise KeyError(f"unknown patch spec {spec!r}")
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i], i))
    rng = random.Random(seed)
    placed: List[Rect] = []
    for idx in order:
        p = int(sizes[idx])
        if image_size % grid != 0 or p % grid != 0:
            raise ValueError(f"patch size {p} / image {image_size} 必须能被 grid={grid} 整除")
        slots = [(x, y) for y in range(0, image_size - p + 1, p) for x in range(0, image_size - p + 1, p)]
        rng.shuffle(slots)
        found = None
        for (x, y) in slots:
            cand = Rect(x, y, p, p)
            if all(not _overlaps(cand, other) for other in placed):
                found = cand
                break
        attempts = 0
        while found is None and attempts < max_tries:
            attempts += 1
            x = rng.randrange(0, image_size - p + 1, grid)
            y = rng.randrange(0, image_size - p + 1, grid)
            cand = Rect(x, y, p, p)
            if all(not _overlaps(cand, other) for other in placed):
                found = cand
        if found is None:
            raise RuntimeError(f"{spec}: 放不下第 {idx} 个 {p}px patch（已放 {len(placed)}/{len(sizes)}）")
        placed.append(found)
    return placed


def _feather_weight(rect: Rect, image_size: int, w_px: float) -> torch.Tensor:
    """单个 patch 的 smoothstep 羽化权重（仅该 patch 外接框内非零）。"""
    p = rect.h
    ys = torch.arange(rect.y, rect.y + rect.h, dtype=torch.float32) - rect.y
    xs = torch.arange(rect.x, rect.x + rect.w, dtype=torch.float32) - rect.x
    dy = torch.minimum(ys, (p - 1) - ys).view(-1, 1)
    dx = torch.minimum(xs, (rect.w - 1) - xs).view(1, -1)
    d = torch.minimum(dy, dx).clamp_min(0.0)
    r = torch.clamp(d / max(float(w_px), 1e-6), 0.0, 1.0)
    return 3.0 * r * r - 2.0 * r * r * r


def rasterize(
    rects: Sequence[Rect],
    *,
    image_size: int = IMAGE_SIZE,
    feather: bool = False,
    feather_width: float | None = None,
    alpha_scale: float = 1.0,
) -> torch.Tensor:
    """把矩形列表画成 (image_size, image_size) 的 [0,1] 掩码。

    ``feather_width`` 缺省为该 patch 边长的 1/8（历史行为）；``alpha_scale``
    用于“等有效注入量”对照（权重整体放大后截断到 1）。
    """
    mask = torch.zeros(image_size, image_size, dtype=torch.float32)
    for rect in rects:
        if feather:
            w_px = rect.w / 8.0 if feather_width is None else float(feather_width)
            weight = _feather_weight(rect, image_size, w_px)
            if alpha_scale != 1.0:
                weight = torch.clamp(weight * float(alpha_scale), 0.0, 1.0)
        else:
            weight = torch.ones(rect.h, rect.w, dtype=torch.float32)
        mask[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w] = torch.maximum(
            mask[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w], weight
        )
    return mask


def coverage(mask: torch.Tensor) -> float:
    """实际权重覆盖率 Σα / N²（B 的 effective_coverage）。"""
    return float(mask.mean().item())


def subband_mask(
    rects: Sequence[Rect],
    *,
    feather: bool = False,
    subband_size: int = SUBBAND_SIZE,
) -> torch.Tensor:
    """把图像坐标 patch 映射到一级 Haar 子带（32×32）上的掩码。"""
    mask = torch.zeros(subband_size, subband_size, dtype=torch.float32)
    for rect in rects:
        x, y, w, h = rect.tokens
        if feather:
            # 羽化在子带分辨率下按 w_sub = (p/8)/2 处理
            w_sub = max(float(w) / 8.0, 1e-6)
            ys = torch.arange(h, dtype=torch.float32)
            xs = torch.arange(w, dtype=torch.float32)
            dy = torch.minimum(ys, (h - 1) - ys).view(-1, 1)
            dx = torch.minimum(xs, (w - 1) - xs).view(1, -1)
            d = torch.minimum(dy, dx).clamp_min(0.0)
            r = torch.clamp(d / w_sub, 0.0, 1.0)
            weight = 3.0 * r * r - 2.0 * r * r * r
        else:
            weight = torch.ones(h, w, dtype=torch.float32)
        mask[y : y + h, x : x + w] = torch.maximum(mask[y : y + h, x : x + w], weight)
    return mask
