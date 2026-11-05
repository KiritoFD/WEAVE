"""DINOv2 patch 匹配（gt-build「局部 patch 匹配」）。

协议与仓库历史评测一致：``utils.compute_dino_metrics.DINO_TRANSFORM``（Resize 224 +
CenterCrop 224），因此 patch token 网格固定 16×16，每 token 覆盖图像 32px。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from tools.pseudo_gt.masks import Rect
from utils.compute_dino_metrics import DINO_TRANSFORM, load_dino

DEVICE = "cuda"
DINO_NAME = "facebook/dinov2-small"


@dataclass
class Match:
    """一个源 patch 与供体 patch 的匹配结果。"""

    src: Rect
    donor: Rect
    q: float
    mutual_ratio: float


def assert_cuda(t: torch.Tensor, what: str = "tensor") -> torch.Tensor:
    if not t.is_cuda:
        raise RuntimeError(f"{what} 不在 cuda 上（device={t.device}），禁止静默 CPU 回退")
    return t


def load_dino_model(device: str = DEVICE, allow_network: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：本实验禁止 CPU 回退")
    # 复用评测缓存里的 HF 快照（ASSET_ROOT/eval_cache/hf），避免离线机器上走网络
    try:
        import paths

        cache_dir = str(paths.eval_cache_root("hf"))
    except Exception:
        cache_dir = str(Path.home() / ".cache" / "huggingface")
    model = load_dino(DINO_NAME, device, cache_dir, allow_network)
    model.eval()
    return model


@torch.inference_mode()
def encode_dino(
    model,
    images: Sequence[Image.Image],
    *,
    batch: int = 16,
    device: str = DEVICE,
    use_fp16: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (CLS [B,384], patch [B,256,384])，全部 fp16 在 cuda 上。"""
    if len(images) == 0:
        raise ValueError("空图像列表")
    cls_out: List[torch.Tensor] = []
    patch_out: List[torch.Tensor] = []
    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        x = torch.stack([DINO_TRANSFORM(im.convert("RGB")) for im in chunk]).to(device)
        assert_cuda(x, "dino input")
        if use_fp16:
            x = x.half()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            hidden = model(x).last_hidden_state
        assert_cuda(hidden, "dino hidden")
        # DINOv2-small: 1 CLS + 4 register + 256 patch = 261；无 register 为 257
        n = hidden.shape[1]
        if n == 261:
            cls, patches = hidden[:, 0], hidden[:, 5:]
        elif n == 257:
            cls, patches = hidden[:, 0], hidden[:, 1:]
        else:
            raise RuntimeError(f"意外的 DINO token 数 {n}（期望 257 或 261）")
        cls_out.append(F.normalize(cls.float(), dim=-1))
        patch_out.append(F.normalize(patches.float(), dim=-1))
    return torch.cat(cls_out, 0), torch.cat(patch_out, 0)


DINO_TOKEN_PX = 32  # 512 图像 → 224 DINO，16×16 token，每 token 32 图像像素


def _rect_tokens(rect: Rect, token_grid: int = 16) -> Tuple[slice, slice]:
    x, y, w, h = (rect.x // DINO_TOKEN_PX, rect.y // DINO_TOKEN_PX,
                  rect.w // DINO_TOKEN_PX, rect.h // DINO_TOKEN_PX)
    if x + w > token_grid or y + h > token_grid:
        raise ValueError(f"patch {rect} 超出 {token_grid}×{token_grid} token 网格")
    return slice(y, y + h), slice(x, x + w)


def match_rect(
    src_tokens: torch.Tensor,
    donor_tokens: torch.Tensor,
    rect: Rect,
    *,
    min_q: float = 0.30,
) -> Optional[Match]:
    """对单个 patch 做双向最近邻 + 全局平移一致性匹配。

    ``src_tokens``/``donor_tokens``: [256, D]，已 L2 归一化。
    """
    sy, sx = _rect_tokens(rect)
    # token 张量是 [256, D]，必须先还原成 16×16 网格再按 (行, 列) 切片
    src = src_tokens.reshape(16, 16, -1)[sy, sx].reshape(-1, src_tokens.shape[-1])  # [T, D]
    if src.shape[0] == 0:
        return None
    donor = donor_tokens.reshape(-1, donor_tokens.shape[-1])           # [256, D]
    sim = src @ donor.t()                                              # [T, 256]
    best_d = sim.argmax(dim=1)
    best_s = sim.argmax(dim=0)
    mutual = best_s[best_d] == torch.arange(src.shape[0], device=src.device)
    ratio = float(mutual.float().mean().item())
    q = float(sim.gather(1, best_d[:, None]).mean().item())
    if q < min_q or ratio < 0.15:
        return None

    # 源 token 在 16×16 网格上的位置
    ys = torch.arange(rect.y // 32, (rect.y + rect.h) // 32, device=src.device)
    xs = torch.arange(rect.x // 32, (rect.x + rect.w) // 32, device=src.device)
    yy = ys[:, None].expand(len(ys), len(xs)).reshape(-1)
    xx = xs[None, :].expand(len(ys), len(xs)).reshape(-1)
    dy = (best_d // 16 - yy)[mutual]
    dx = (best_d % 16 - xx)[mutual]
    if dy.numel() == 0:
        return None
    shift_y = int(dy.float().median().item())
    shift_x = int(dx.float().median().item())
    # 统一在 32px 的 DINO token 单位里算位移（rect.tokens 是 16px 子带单位，不能混用）
    x32, y32 = rect.x // DINO_TOKEN_PX, rect.y // DINO_TOKEN_PX
    w32, h32 = rect.w // DINO_TOKEN_PX, rect.h // DINO_TOKEN_PX
    nx = max(0, min(x32 + shift_x, 16 - w32))
    ny = max(0, min(y32 + shift_y, 16 - h32))
    donor_rect = Rect(nx * DINO_TOKEN_PX, ny * DINO_TOKEN_PX,
                      w32 * DINO_TOKEN_PX, h32 * DINO_TOKEN_PX)
    return Match(src=rect, donor=donor_rect, q=q, mutual_ratio=ratio)


def match_layout(
    src_tokens: torch.Tensor,
    donor_tokens: torch.Tensor,
    rects: Sequence[Rect],
    *,
    min_q: float = 0.30,
    min_good_frac: float = 0.75,
) -> Tuple[Optional[List[Match]], float]:
    """整张图的匹配。返回 (matches, 平均 q)。

    逐 patch 记录 q_i；只有互近邻一致性不过关的 patch 被丢弃（不下发坐标）。
    若可用 patch 低于 ``min_good_frac``，或高质量 patch（q≥min_q）占比低于
    ``min_good_frac``，则整图判为失败（gt-practice：不允许用低质量匹配凑数量）。
    少量弱 patch 保留（同一批匹配在 A–E 之间共享，噪声对比较是共同的）。
    """
    out: List[Match] = []
    qs: List[float] = []
    good = 0
    for rect in rects:
        m = match_rect(src_tokens, donor_tokens, rect, min_q=-2.0)
        if m is None:
            continue
        out.append(m)
        qs.append(m.q)
        if m.q >= min_q and m.mutual_ratio >= 0.15:
            good += 1
    if not out or len(out) / len(rects) < min_good_frac or good / len(rects) < min_good_frac:
        return None, 0.0
    return out, sum(qs) / len(qs)


@torch.inference_mode()
def retrieve_topk(
    cls_matrix: torch.Tensor,
    query_idx: int,
    *,
    allowed: torch.Tensor,
    k: int = 8,
) -> List[int]:
    """按 CLS 余弦检索 top-k（``allowed`` 为候选下标集合）。"""
    sim = cls_matrix[query_idx] @ cls_matrix.t()
    sim = sim.clone()
    sim[~allowed] = -2.0
    sim[query_idx] = -2.0
    return torch.topk(sim, k=min(k, int(allowed.sum()))).indices.tolist()
