"""伪GT 构造器 A/B/C/D/D-HF/E（gt-build §2）。

所有构造都在 512 分辨率或 64×64 latent 上完成；RGB 路径由调用方 VAE mean 编码。
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F

from tools.pseudo_gt.matching import Match
from tools.pseudo_gt.masks import Rect, rasterize, subband_mask
from wavelet import dwt2_haar, idwt2_haar


def _patch_alpha(rects: Sequence[Rect], feather: bool, image_size: int = 512) -> torch.Tensor:
    return rasterize(rects, image_size=image_size, feather=feather)


def construct_A(src: torch.Tensor, donor: torch.Tensor, matches: Sequence[Match]) -> torch.Tensor:
    """硬替换：把供体 patch 逐像素贴到源的对应位置（无羽化）。"""
    out = src.clone()
    for m in matches:
        out[:, :, m.src.y : m.src.y + m.src.h, m.src.x : m.src.x + m.src.w] = donor[
            :, :, m.donor.y : m.donor.y + m.donor.h, m.donor.x : m.donor.x + m.donor.w
        ]
    return out


def construct_B(src: torch.Tensor, donor: torch.Tensor, matches: Sequence[Match]) -> torch.Tensor:
    """羽化混合：alpha = smoothstep(距离 / (p/8))。"""
    src_rects = [m.src for m in matches]
    alpha = _patch_alpha(src_rects, feather=True, image_size=src.shape[-1]).to(src.device)[None, None]
    donor_placed = src.clone()
    for m in matches:
        donor_placed[:, :, m.src.y : m.src.y + m.src.h, m.src.x : m.src.x + m.src.w] = donor[
            :, :, m.donor.y : m.donor.y + m.donor.h, m.donor.x : m.donor.x + m.donor.w
        ]
    return src * (1 - alpha) + donor_placed * alpha


def construct_E(corrupted_src: torch.Tensor, donor: torch.Tensor, matches: Sequence[Match], *, feather: bool) -> torch.Tensor:
    """反向污染：向真实输入图贴入外来 patch（Feather 或 Hard）。"""
    return construct_B(corrupted_src, donor, matches) if feather else construct_A(corrupted_src, donor, matches)


def _paste_band(base: torch.Tensor, donor: torch.Tensor, matches: Sequence[Match]) -> torch.Tensor:
    """把供体子带在 donor 位置的内容搬到源图的 src 位置。"""
    out = base.clone()
    for m in matches:
        sx, sy, sw, sh = m.src.tokens
        dx, dy, _, _ = m.donor.tokens
        out[:, :, sy : sy + sh, sx : sx + sw] = donor[:, :, dy : dy + sh, dx : dx + sw]
    return out


def _subband_alpha(rects: Sequence[Rect], feather: bool) -> torch.Tensor:
    return subband_mask(rects, feather=feather)  # [32,32]


def _donor_rects(matches: Sequence[Match]) -> List[Rect]:
    return [m.donor for m in matches]


def construct_C(z_src: torch.Tensor, z_donor: torch.Tensor, matches: Sequence[Match]) -> torch.Tensor:
    """只换 LH/HL：LL 与 HH 严格保持源图。"""
    ll_s, lh_s, hl_s, hh_s = dwt2_haar(z_src)
    _, lh_d, hl_d, _ = dwt2_haar(z_donor)
    lh = _paste_band(lh_s, lh_d, matches)
    hl = _paste_band(hl_s, hl_d, matches)
    return idwt2_haar(ll_s, lh, hl, hh_s)


def construct_D(
    z_src: torch.Tensor,
    z_donor: torch.Tensor,
    matches: Sequence[Match],
    *,
    ratio_clip: tuple[float, float] = (0.5, 2.0),
    keep_ll: bool = True,
) -> torch.Tensor:
    """局部 AdaIN 统计迁移（LL/LH/HL；``keep_ll=False`` 时即 D-HF）。"""
    ll_s, lh_s, hl_s, hh_s = dwt2_haar(z_src)
    ll_d, lh_d, hl_d, _ = dwt2_haar(z_donor)
    bands = {}
    for name, s, d, use in (("LL", ll_s, ll_d, keep_ll), ("LH", lh_s, lh_d, True), ("HL", hl_s, hl_d, True)):
        if not use:
            bands[name] = s
            continue
        bands[name] = s.clone()
        for m in matches:
            sx, sy, sw, sh = m.src.tokens
            dx, dy, _, _ = m.donor.tokens
            mu_s = s[:, :, sy : sy + sh, sx : sx + sw].mean(dim=(-1, -2), keepdim=True)
            mu_d = d[:, :, dy : dy + sh, dx : dx + sw].mean(dim=(-1, -2), keepdim=True)
            var_s = s[:, :, sy : sy + sh, sx : sx + sw].var(dim=(-1, -2), keepdim=True, unbiased=False)
            var_d = d[:, :, dy : dy + sh, dx : dx + sw].var(dim=(-1, -2), keepdim=True, unbiased=False)
            ratio = torch.sqrt(var_d + 1e-6) / torch.sqrt(var_s + 1e-6)
            ratio = torch.clamp(ratio, ratio_clip[0], ratio_clip[1])
            transferred = (s[:, :, sy : sy + sh, sx : sx + sw] - mu_s) * ratio + mu_d
            bands[name][:, :, sy : sy + sh, sx : sx + sw] = transferred
    return idwt2_haar(bands["LL"], bands["LH"], bands["HL"], hh_s)


def seam_gradient(image: torch.Tensor, rects: Sequence[Rect]) -> float:
    """patch 边界 1px 环的平均梯度 / 图像内部平均梯度。"""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    gray = image.mean(dim=1, keepdim=True)
    gy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
    gx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    g = torch.zeros_like(gray)
    g[:, :, 1:, :] += gy
    g[:, :, :, 1:] += gx
    boundary = torch.zeros_like(gray)
    inner = torch.ones_like(gray)
    for r in rects:
        boundary[:, :, r.y : r.y + r.h, max(r.x - 1, 0) : r.x + 1] = 1
        boundary[:, :, r.y : r.y + r.h, r.x + r.w - 1 : min(r.x + r.w + 1, 512)] = 1
        boundary[:, :, max(r.y - 1, 0) : r.y + 1, r.x : r.x + r.w] = 1
        boundary[:, :, r.y + r.h - 1 : min(r.y + r.h + 1, 512), r.x : r.x + r.w] = 1
        inner[:, :, r.y + 2 : r.y + r.h - 2, r.x + 2 : r.x + r.w - 2] = 0
    b = float((g * boundary).sum() / boundary.sum().clamp_min(1.0))
    i = float((g * inner).sum() / inner.sum().clamp_min(1.0))
    return b / max(i, 1e-6)


def band_energy_share(z_src: torch.Tensor, z_pseudo: torch.Tensor) -> dict[str, float]:
    """差分能量在 LL/LH/HL/HH 上的占比。"""
    ll_s, lh_s, hl_s, hh_s = dwt2_haar(z_src)
    ll_p, lh_p, hl_p, hh_p = dwt2_haar(z_pseudo)
    energies = {
        "LL": float(((ll_p - ll_s) ** 2).sum().item()),
        "LH": float(((lh_p - lh_s) ** 2).sum().item()),
        "HL": float(((hl_p - hl_s) ** 2).sum().item()),
        "HH": float(((hh_p - hh_s) ** 2).sum().item()),
    }
    total = sum(energies.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in energies.items()}
