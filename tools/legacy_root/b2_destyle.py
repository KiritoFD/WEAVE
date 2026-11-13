"""路线 B2：真实绘画的严格对齐自监督配对（B1 的预训练阶段）。

核心构造：

    x' = Destyle(y)  # 只做颜色归一化 + 低频外观扰动 + 受控高频抑制，不改几何
    r_s = MixRef(y_ref_other_same_style)  # 同风格另一张画，裁块/位置打乱/混合同风格作品
    T_phi(x', r_s) -> y

Destyle 的几何不变性：不 warp、不裁剪 x'、不 resize，只做逐像素光度变换与
低通噪声；高频抑制使用"低通 + alpha * 高频"的重建式写法，边缘位置不动。

参考图防复制：参考图只提供"操作集合"。本文件在 **token/块级别** 打乱参考图，
并支持混入同风格其他作品；B1 的 attention 本身对 token 顺序置换不变，因此
参考图的空间布局不会进入输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from wavelet import dwt2_haar, idwt2_haar


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class DestyleConfig:
    """Destyle 的随机化范围。默认参数对应"强去除风格但保留几何"。"""

    # 1) 随机颜色归一化
    color_prob: float = 0.9
    color_mode_probs: Tuple[float, float, float] = (0.45, 0.35, 0.20)  # 灰度 / 逐通道 / 全局对比
    color_shift: float = 0.35          # 均值抖动幅度（相对值）
    color_scale: float = 0.6           # 标准差缩放范围 ±
    # 2) 低频外观扰动
    lowfreq_prob: float = 0.9
    lowfreq_grid: int = 4
    lowfreq_amp: float = 0.25
    # 3) 受控高频抑制
    hf_suppress_prob: float = 0.9
    hf_suppress_min: float = 0.05      # 保留的高频比例下限
    hf_suppress_max: float = 0.45      # 保留的高频比例上限
    blur_prob: float = 0.5
    blur_sigma: Tuple[float, float] = (0.5, 1.5)
    # 图像空间输出范围；latent 空间传 None
    clamp_range: Tuple[float, float] | None = (0.0, 1.0)
    extra: Dict = field(default_factory=dict)


@dataclass
class ReferenceMixConfig:
    """参考图打乱 / 裁块 / 混合作品的概率与强度。"""

    patch_size: int = 4
    shuffle_prob: float = 0.9
    crop_prob: float = 0.5
    crop_scale: Tuple[float, float] = (0.5, 1.0)
    mix_prob: float = 0.5
    mix_ratio: Tuple[float, float] = (0.2, 0.6)
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 随机工具
# ---------------------------------------------------------------------------

def _gen(generator: torch.Generator | None) -> torch.Generator:
    return generator if generator is not None else torch.default_generator


def _rand(generator: torch.Generator | None, device: torch.device) -> float:
    return float(torch.rand(1, generator=_gen(generator), device=device))


def _randint(high: int, generator: torch.Generator | None, device: torch.device) -> int:
    return int(torch.randint(high, (1,), generator=_gen(generator), device=device))


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """可分离高斯模糊（reflect padding），保持几何位置。"""
    if sigma <= 0:
        return x
    C = x.shape[1]
    r = max(1, min(int(3.0 * sigma), min(x.shape[-1], x.shape[-2]) - 1))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / float(sigma)).pow(2))
    k = k / k.sum()
    wx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    wy = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), wx, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), wy, groups=C)
    return x


def _luminance(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] >= 3:
        w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x[:, :3] * w).sum(dim=1, keepdim=True)
    return x.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Destyle：图像空间
# ---------------------------------------------------------------------------

def destyle_image(
    y: torch.Tensor,
    cfg: DestyleConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """x' = Destyle(y)。只改光度，不改几何。

    Args:
        y: (B,C,H,W)
    Returns:
        同形状 tensor。
    """
    cfg = cfg or DestyleConfig()
    out = y.float().clone()
    B, C, H, W = out.shape
    dev = out.device

    # 1) 随机颜色归一化
    if _rand(generator, dev) < cfg.color_prob:
        mode_p = torch.tensor(cfg.color_mode_probs, device=dev)
        mode = int(torch.multinomial(mode_p / mode_p.sum(), 1, generator=_gen(generator)))
        if mode == 0:
            # 去色：只保留亮度，再加轻微随机染色
            g = _luminance(out)
            tint = 1.0 + (torch.rand(1, C, 1, 1, generator=_gen(generator), device=dev) - 0.5)
            out = g * tint
        elif mode == 1:
            # 逐通道标准化 + 随机目标均值/方差
            mean = out.mean(dim=(2, 3), keepdim=True)
            std = out.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
            new_mean = mean + (torch.rand(B, C, 1, 1, generator=_gen(generator), device=dev) - 0.5) * (2 * cfg.color_shift)
            new_std = std * (1.0 + (torch.rand(B, C, 1, 1, generator=_gen(generator), device=dev) - 0.5) * 2 * cfg.color_scale)
            out = (out - mean) / std * new_std + new_mean
        else:
            # 全局亮度/对比度抖动
            g = _luminance(out)
            gm = g.mean(dim=(2, 3), keepdim=True)
            gs = g.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
            contrast = 1.0 + (torch.rand(1, 1, 1, 1, generator=_gen(generator), device=dev) - 0.5) * 2 * cfg.color_scale
            out = (out - gm) * contrast + gm

    # 2) 低频外观扰动：粗网格随机场双线性放大
    if _rand(generator, dev) < cfg.lowfreq_prob:
        g = cfg.lowfreq_grid
        noise = torch.randn(B, C, g, g, generator=_gen(generator), device=dev)
        field = F.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False)
        out = out + field * cfg.lowfreq_amp

    # 3) 受控高频抑制（重建式：低通 + alpha*高频，边缘位置不变）
    if _rand(generator, dev) < cfg.blur_prob:
        sigma = cfg.blur_sigma[0] + _rand(generator, dev) * (cfg.blur_sigma[1] - cfg.blur_sigma[0])
    else:
        sigma = 0.5
    lp = _gaussian_blur(out, sigma)
    hf = out - lp
    if _rand(generator, dev) < cfg.hf_suppress_prob:
        alpha = cfg.hf_suppress_min + _rand(generator, dev) * (cfg.hf_suppress_max - cfg.hf_suppress_min)
    else:
        alpha = 1.0
    out = lp + alpha * hf

    if cfg.clamp_range is not None:
        out = out.clamp(*cfg.clamp_range)
    return out.to(dtype=y.dtype)


# ---------------------------------------------------------------------------
# Destyle：latent 空间（训练 T_phi 直接用 latent，避免训练 step 内跑 VAE）
# ---------------------------------------------------------------------------

def destyle_latent(
    z: torch.Tensor,
    cfg: DestyleConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """latent 版 Destyle：逐通道仿射 + 低频扰动 + Haar HF 衰减，无几何变换。"""
    cfg = cfg or DestyleConfig()
    zf = z.float()
    B, C, H, W = zf.shape
    dev = zf.device
    out = zf.clone()

    if _rand(generator, dev) < cfg.color_prob:
        mean = out.mean(dim=(2, 3), keepdim=True)
        std = out.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        new_mean = mean + (torch.rand(B, C, 1, 1, generator=_gen(generator), device=dev) - 0.5) * (2 * cfg.color_shift)
        new_std = std * (1.0 + (torch.rand(B, C, 1, 1, generator=_gen(generator), device=dev) - 0.5) * 2 * cfg.color_scale)
        out = (out - mean) / std * new_std + new_mean

    if _rand(generator, dev) < cfg.lowfreq_prob:
        noise = torch.randn(B, C, cfg.lowfreq_grid, cfg.lowfreq_grid, generator=_gen(generator), device=dev)
        field = F.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False)
        out = out + field * cfg.lowfreq_amp

    if _rand(generator, dev) < cfg.hf_suppress_prob:
        alpha = cfg.hf_suppress_min + _rand(generator, dev) * (cfg.hf_suppress_max - cfg.hf_suppress_min)
    else:
        alpha = 1.0
    ll, lh, hl, hh = dwt2_haar(out)
    out = idwt2_haar(ll, lh * alpha, hl * alpha, hh * alpha)
    return out.to(dtype=z.dtype)


# ---------------------------------------------------------------------------
# 参考图块级扰动
# ---------------------------------------------------------------------------

def shuffle_patches(
    x: torch.Tensor,
    patch_size: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """按 patch 网格随机打乱空间位置；patch 集合完全保留。

    这是"参考图不提供空间相位"的最强形式：只改变排列，不改变内容。
    """
    B, C, H, W = x.shape
    p = int(patch_size)
    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    xp = F.pad(x.float(), (0, pad_w, 0, pad_h), mode="replicate")
    Hp, Wp = xp.shape[-2:]
    patches = F.unfold(xp, kernel_size=p, stride=p)  # (B, C*p*p, N)
    n = patches.shape[-1]
    perm = torch.randperm(n, generator=_gen(generator), device=x.device)
    patches = patches[:, :, perm]
    out = F.fold(patches, output_size=(Hp, Wp), kernel_size=p, stride=p)
    return out[..., :H, :W].to(dtype=x.dtype)


def crop_resize(
    x: torch.Tensor,
    patch_size: int = 4,
    scale: Tuple[float, float] = (0.5, 1.0),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """随机裁剪参考图再缩回原尺寸（破坏物体尺度/布局，允许用于参考图）。"""
    B, C, H, W = x.shape
    s = scale[0] + _rand(generator, x.device) * (scale[1] - scale[0])
    ch = max(int(patch_size), int(H * s))
    cw = max(int(patch_size), int(W * s))
    top = _randint(H - ch + 1, generator, x.device)
    left = _randint(W - cw + 1, generator, x.device)
    crop = x[..., top : top + ch, left : left + cw]
    return F.interpolate(crop.float(), size=(H, W), mode="bilinear", align_corners=False).to(dtype=x.dtype)


def mix_patches(
    x: torch.Tensor,
    donor: torch.Tensor,
    patch_size: int = 4,
    ratio: float = 0.4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """从同风格另一张作品混入一部分 patch（donor patch 也会被打乱）。"""
    if x.shape != donor.shape:
        donor = F.interpolate(donor.float(), size=x.shape[-2:], mode="bilinear", align_corners=False).to(dtype=x.dtype)
    B, C, H, W = x.shape
    p = int(patch_size)
    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    xp = F.pad(x.float(), (0, pad_w, 0, pad_h), mode="replicate")
    dp = F.pad(donor.float(), (0, pad_w, 0, pad_h), mode="replicate")
    Hp, Wp = xp.shape[-2:]
    px = F.unfold(xp, kernel_size=p, stride=p)  # (B,Cp,N)
    pd = F.unfold(dp, kernel_size=p, stride=p)
    n = px.shape[-1]
    donor_perm = torch.randperm(n, generator=_gen(generator), device=x.device)
    pd = pd[:, :, donor_perm]
    mask = (torch.rand(B, 1, n, generator=_gen(generator), device=x.device) < float(ratio)).to(px.dtype)
    mixed = px * (1.0 - mask) + pd * mask
    out = F.fold(mixed, output_size=(Hp, Wp), kernel_size=p, stride=p)
    return out[..., :H, :W].to(dtype=x.dtype)


class ReferenceMixer:
    """把参考图变成"只含操作、不含空间相位"的视图。"""

    def __init__(self, cfg: ReferenceMixConfig | None = None) -> None:
        self.cfg = cfg or ReferenceMixConfig()

    def __call__(
        self,
        reference: torch.Tensor,
        donor: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, bool]]:
        cfg = self.cfg
        out = reference
        meta = {"mixed": False, "cropped": False, "shuffled": False}
        if donor is not None and _rand(generator, reference.device) < cfg.mix_prob:
            ratio = cfg.mix_ratio[0] + _rand(generator, reference.device) * (cfg.mix_ratio[1] - cfg.mix_ratio[0])
            out = mix_patches(out, donor, cfg.patch_size, ratio, generator)
            meta["mixed"] = True
        if _rand(generator, reference.device) < cfg.crop_prob:
            out = crop_resize(out, cfg.patch_size, cfg.crop_scale, generator)
            meta["cropped"] = True
        if _rand(generator, reference.device) < cfg.shuffle_prob:
            out = shuffle_patches(out, cfg.patch_size, generator)
            meta["shuffled"] = True
        return out, meta


# ---------------------------------------------------------------------------
# B2 配对构造
# ---------------------------------------------------------------------------

def build_image_pair(
    target: torch.Tensor,
    reference: torch.Tensor,
    donor: torch.Tensor | None = None,
    *,
    destyle_cfg: DestyleConfig | None = None,
    mix_cfg: ReferenceMixConfig | None = None,
    generator: torch.Generator | None = None,
) -> Dict[str, torch.Tensor]:
    """图像空间配对：(x', r_s, y)。

    x' 与 y 严格同几何；r_s 是参考图（可为同风格另一张画或混合作品）。
    """
    mixer = ReferenceMixer(mix_cfg)
    xp = destyle_image(target, destyle_cfg, generator)
    rs, meta = mixer(reference, donor, generator)
    return {
        "content": xp,
        "reference": rs,
        "target": target.float(),
        "meta": meta,
    }


def build_latent_pair(
    target_latent: torch.Tensor,
    reference_latent: torch.Tensor,
    donor_latent: torch.Tensor | None = None,
    *,
    destyle_cfg: DestyleConfig | None = None,
    mix_cfg: ReferenceMixConfig | None = None,
    generator: torch.Generator | None = None,
) -> Dict[str, torch.Tensor]:
    """latent 空间配对：(x', r_s, y)，供 T_phi 阶段 1 直接使用。"""
    mixer = ReferenceMixer(mix_cfg)
    xp = destyle_latent(target_latent, destyle_cfg, generator)
    rs, meta = mixer(reference_latent, donor_latent, generator)
    return {
        "content_latent": xp,
        "reference_latent": rs,
        "target_latent": target_latent.float(),
        "meta": meta,
    }


def lowpass_geometry_correlation(a: torch.Tensor, b: torch.Tensor, patch: int = 4) -> float:
    """几何不变性诊断：低通亮度图的相关性。用于验证 Destyle 没有搬几何。"""
    ga = _luminance(a.float())
    gb = _luminance(b.float())
    ga = F.avg_pool2d(ga, patch, patch)
    gb = F.avg_pool2d(gb, patch, patch)
    xa = ga.flatten(1)
    xb = gb.flatten(1)
    xa = xa - xa.mean(dim=-1, keepdim=True)
    xb = xb - xb.mean(dim=-1, keepdim=True)
    denom = xa.pow(2).sum(-1).sqrt() * xb.pow(2).sum(-1).sqrt()
    return float(((xa * xb).sum(-1) / denom.clamp_min(1e-8)).mean())


__all__ = [
    "DestyleConfig",
    "ReferenceMixConfig",
    "destyle_image",
    "destyle_latent",
    "shuffle_patches",
    "crop_resize",
    "mix_patches",
    "ReferenceMixer",
    "build_image_pair",
    "build_latent_pair",
    "lowpass_geometry_correlation",
]
