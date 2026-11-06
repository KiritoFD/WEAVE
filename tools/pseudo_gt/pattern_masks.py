"""空间 Pattern 掩码：8 组对照 + 位置/覆盖率消融的网格选择与掩码测量。

所有 Pattern 都定义在 64px 网格（8×8 = 64 格）或 128px 网格（4×4 = 16 格）上，
只能从“已通过内容匹配”的格子中选（q≥0.30 且互近邻≥0.15），不为凑形状放宽阈值。
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

IMAGE = 512
Q_MIN = 0.30
MUTUAL_MIN = 0.15

#: 8 组主 Pattern；k = patch 数，size = 边长（px）
PATTERN_SPECS: Dict[str, dict] = {
    "R64": {"size": 64, "k": 16, "kind": "random"},
    "G64": {"size": 64, "k": 16, "kind": "grid"},
    "C64": {"size": 64, "k": 16, "kind": "cluster"},
    "S64": {"size": 64, "k": 16, "kind": "spiral"},
    "F64": {"size": 64, "k": 16, "kind": "field"},
    "R128": {"size": 128, "k": 4, "kind": "random"},
    "G128": {"size": 128, "k": 4, "kind": "grid"},
    "C128": {"size": 128, "k": 4, "kind": "cluster"},
}

#: 位置/覆盖率消融（固定 64px）
ABLATION_SPECS: Dict[str, dict] = {
    "L0": {"size": 64, "k": 16, "kind": "random"},
    "L1": {"size": 64, "k": 16, "kind": "texture_high"},
    "L2": {"size": 64, "k": 16, "kind": "texture_low"},
    "L3": {"size": 64, "k": 16, "kind": "saliency_high"},
    "L4": {"size": 64, "k": 16, "kind": "quality_high"},
    "C12": {"size": 64, "k": 8, "kind": "random"},
    "C25": {"size": 64, "k": 16, "kind": "random"},
    "C37": {"size": 64, "k": 24, "kind": "random"},
    "C50": {"size": 64, "k": 32, "kind": "random"},
}


def cells_of(size: int) -> int:
    return IMAGE // size


def nominal_coverage(spec: str) -> float:
    """名义覆盖率 = (patch 边长² × patch 数) / 512²。"""
    s = PATTERN_SPECS.get(spec) or ABLATION_SPECS[spec]
    return float(s["size"] ** 2 * s["k"] / (IMAGE ** 2))


def rects_from_cells(cells: Sequence[Tuple[int, int]], size: int):
    from tools.pseudo_gt.masks import Rect

    return [Rect(cx * size, cy * size, size, size) for cx, cy in cells]


def _qualified(qmap: np.ndarray, mmap: np.ndarray) -> np.ndarray:
    return (qmap >= Q_MIN) & (mmap >= MUTUAL_MIN) & np.isfinite(qmap)


def grid_cells() -> List[Tuple[int, int]]:
    n = cells_of(64)
    return [(x, y) for y in range(n) for x in range(n)]


def grid_cells128() -> List[Tuple[int, int]]:
    n = cells_of(128)
    return [(x, y) for y in range(n) for x in range(n)]


def spiral_order(n: int, rng: random.Random) -> List[Tuple[int, int]]:
    """阿基米德螺旋在 n×n 网格上的路径顺序（随机旋转/镜像/相位）。"""
    total = n * n
    theta0 = rng.uniform(0, 2 * math.pi)
    mirror = -1.0 if rng.random() < 0.5 else 1.0
    center = (n - 1) / 2.0
    r_max = math.sqrt(2) * (n / 2.0)
    steps = max(total * 12, 256)
    seen: List[Tuple[int, int]] = []
    seen_set = set()
    for i in range(steps):
        t = i / (steps - 1)
        r = r_max * t
        th = theta0 + 4.0 * math.pi * t
        x = center + mirror * r * math.cos(th)
        y = center + r * math.sin(th)
        cx, cy = int(round(x - 0.5)), int(round(y - 0.5))
        if 0 <= cx < n and 0 <= cy < n and (cx, cy) not in seen_set:
            seen_set.add((cx, cy))
            seen.append((cx, cy))
    for cy in range(n):
        for cx in range(n):
            if (cx, cy) not in seen_set:
                seen.append((cx, cy))
    return seen[:total]


def field_order(n: int, rng: random.Random) -> List[Tuple[int, int]]:
    """低频随机场（4×4 隐变量双线性放大到 n×n）的响应降序。"""
    base = np.array([[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(4)])
    ys = np.linspace(0, 3, n)
    xs = np.linspace(0, 3, n)
    field = np.empty((n, n), dtype=np.float64)
    for iy, y in enumerate(ys):
        y0, y1 = int(math.floor(y)), min(int(math.floor(y)) + 1, 3)
        fy = y - y0
        for ix, x in enumerate(xs):
            x0, x1 = int(math.floor(x)), min(int(math.floor(x)) + 1, 3)
            fx = x - x0
            field[iy, ix] = (
                base[y0, x0] * (1 - fx) * (1 - fy)
                + base[y0, x1] * fx * (1 - fy)
                + base[y1, x0] * (1 - fx) * fy
                + base[y1, x1] * fx * fy
            )
    order = np.argsort(-field, axis=None)
    return [(int(j % n), int(j // n)) for j in order]


def select_cells(
    spec: str,
    *,
    qmap: np.ndarray,
    mmap: np.ndarray,
    texture: Optional[np.ndarray] = None,
    saliency: Optional[np.ndarray] = None,
    rng: Optional[random.Random] = None,
    max_tries: int = 24,
) -> Optional[List[Tuple[int, int]]]:
    """按 spec 选择格子；任何格子不合格就不选，凑不齐则返回 None。"""
    all_specs = {**PATTERN_SPECS, **ABLATION_SPECS}
    meta = all_specs[spec]
    size, k, kind = meta["size"], int(meta["k"]), meta["kind"]
    n = cells_of(size)
    assert qmap.shape == (n, n) and mmap.shape == (n, n), f"{spec}: 网格 {qmap.shape} != {n}×{n}"
    qual = _qualified(qmap, mmap)
    flat_q = qmap.reshape(-1)

    def take(order: Sequence[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        out = [c for c in order if qual[c[1], c[0]]]
        if len(out) < k:
            return None
        return out[:k]

    rng = rng or random.Random(0)
    cells = grid_cells() if n == 8 else grid_cells128()
    if kind == "random":
        pool = [c for c in cells if qual[c[1], c[0]]]
        if len(pool) < k:
            return None
        return rng.sample(pool, k)
    if kind == "grid":
        fixed = [(x, y) for y in range(0, n, 2) for x in range(0, n, 2)]
        fixed = fixed[:k]
        return fixed if all(qual[y, x] for x, y in fixed) else None
    if kind == "cluster":
        side = int(round(math.sqrt(k)))
        best = None
        for by in range(n - side + 1):
            for bx in range(n - side + 1):
                block = [(bx + i, by + j) for j in range(side) for i in range(side)]
                if all(qual[y, x] for x, y in block):
                    score = float(np.mean([qmap[y, x] for x, y in block]))
                    if best is None or score > best[0]:
                        best = (score, block)
        return best[1] if best else None
    if kind == "spiral":
        order = spiral_order(n, rng)
        return take(order)
    if kind == "field":
        order = field_order(n, rng)
        return take(order)
    if kind in {"texture_high", "texture_low", "saliency_high", "quality_high"}:
        if kind.startswith("texture"):
            assert texture is not None, "texture 缺失"
            score = texture
            sign = -1.0 if kind == "texture_high" else 1.0
        elif kind == "saliency_high":
            assert saliency is not None, "saliency 缺失"
            score, sign = saliency, -1.0
        else:
            score, sign = qmap, -1.0
        order = sorted(cells, key=lambda c: sign * float(score[c[1], c[0]]))
        return take(order)
    raise KeyError(f"未知 kind={kind}")


def _mask_tensor(cells: Sequence[Tuple[int, int]], size: int, image: int = IMAGE) -> torch.Tensor:
    m = torch.zeros(image, image, dtype=torch.bool)
    for cx, cy in cells:
        m[cy * size : (cy + 1) * size, cx * size : (cx + 1) * size] = True
    return m


def boundary_length(cells: Sequence[Tuple[int, int]], size: int, image: int = IMAGE) -> int:
    """并集边界像素数（含贴图像边的外侧边界）。"""
    m = _mask_tensor(cells, size, image)
    b = 0
    b += int((m[1:, :] != m[:-1, :]).sum()) + int(m[0, :].sum()) + int(m[-1, :].sum())
    b += int((m[:, 1:] != m[:, :-1]).sum()) + int(m[:, 0].sum()) + int(m[:, -1].sum())
    return b


def component_count(cells: Sequence[Tuple[int, int]], size: int) -> int:
    n = cells_of(size)
    grid = np.zeros((n, n), dtype=bool)
    for cx, cy in cells:
        grid[cy, cx] = True
    seen = np.zeros_like(grid)
    count = 0
    for y in range(n):
        for x in range(n):
            if grid[y, x] and not seen[y, x]:
                count += 1
                stack = [(x, y)]
                seen[y, x] = True
                while stack:
                    cx, cy = stack.pop()
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < n and 0 <= ny < n and grid[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
    return count


def mean_patch_distance(cells: Sequence[Tuple[int, int]], size: int) -> float:
    centers = np.array([((cx + 0.5) * size, (cy + 0.5) * size) for cx, cy in cells], dtype=np.float64)
    if len(centers) < 2:
        return 0.0
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(centers), k=1)
    return float(d[iu].mean() / (IMAGE * math.sqrt(2)))


def spatial_periodicity(cells: Sequence[Tuple[int, int]], size: int) -> float:
    """网格自相关在非零滞后上的最大归一化峰（规则网格 → 接近 1）。"""
    n = cells_of(size)
    grid = np.zeros((n, n), dtype=np.float64)
    for cx, cy in cells:
        grid[cy, cx] = 1.0
    if grid.sum() == 0:
        return 0.0
    g = grid - grid.mean()
    denom = float((g * g).sum())
    if denom <= 1e-12:
        return 0.0
    best = 0.0
    for lag in (1, 2, 3):
        if n - lag <= 0:
            continue
        for dy, dx in ((lag, 0), (0, lag), (lag, lag)):
            if dx and n - dx <= 0:
                continue
            a = g[: n - dy, : n - dx]
            b = g[dy:, dx:]
            if a.size == 0:
                continue
            best = max(best, float((a * b).sum() / denom))
    return best


def salient_region_fraction(cells: Sequence[Tuple[int, int]], size: int, saliency: np.ndarray) -> float:
    """替换面积落在显著度前 25% 格子中的比例。"""
    n = cells_of(size)
    sal = saliency if saliency.shape == (n, n) else _pool_to(saliency, n)
    top = np.zeros((n, n), dtype=bool)
    k = max(1, int(round(n * n * 0.25)))
    idx = np.argsort(-sal, axis=None)[:k]
    top[idx // n, idx % n] = True
    hit = sum(1 for cx, cy in cells if top[cy, cx])
    return hit / max(len(cells), 1)


def _pool_to(sal: np.ndarray, n: int) -> np.ndarray:
    if sal.shape == (n, n):
        return sal
    m = sal.shape[0] // n
    return sal.reshape(n, m, n, m).mean(axis=(1, 3))


def mask_metrics(
    cells: Sequence[Tuple[int, int]],
    size: int,
    *,
    qmap: np.ndarray,
    saliency: np.ndarray,
) -> dict:
    """每个 mask 必存的 8 项测量。"""
    q = np.array([qmap[cy, cx] for cx, cy in cells], dtype=np.float64)
    cov = len(cells) * size * size / (IMAGE ** 2)
    quad = {}
    for name, (x0, x1, y0, y1) in {
        "q_tl": (0, IMAGE // 2, 0, IMAGE // 2),
        "q_tr": (IMAGE // 2, IMAGE, 0, IMAGE // 2),
        "q_bl": (0, IMAGE // 2, IMAGE // 2, IMAGE),
        "q_br": (IMAGE // 2, IMAGE, IMAGE // 2, IMAGE),
    }.items():
        inside = sum(
            1 for cx, cy in cells
            if cx * size < x1 and (cx + 1) * size > x0 and cy * size < y1 and (cy + 1) * size > y0
        )
        quad[name] = inside / len(cells) if cells else 0.0
    return {
        "coverage_actual": cov,
        "component_count": component_count(cells, size),
        "boundary_length": boundary_length(cells, size),
        "mean_patch_distance": mean_patch_distance(cells, size),
        "quadrant_coverage": quad,
        "spatial_periodicity": spatial_periodicity(cells, size),
        "match_q_mean": float(q.mean()) if len(q) else 0.0,
        "salient_region_fraction": salient_region_fraction(cells, size, saliency),
    }
