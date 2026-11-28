"""Zero-DINO Training Losses for V36 Reference E-Core (VGG-16 Gram + Patch-SWD + Oklab + Haar DWT).

Strictly Zero DINO, Zero CSD, Zero CLIP, Zero LPIPS in all loss paths.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from exp.v35_cleave_one.losses import FrozenVGGStyleCritic
from src.utils import decode_s10
from src.utils.runtime import bf16_autocast
from src.utils.structure import dwt2_haar, rgb_to_gray, rgb_to_oklab


def _decode_s10_diff(s10_dec: nn.Module, z_scaled: torch.Tensor) -> torch.Tensor:
    dec_dtype = next(s10_dec.parameters()).dtype
    return decode_s10(
        s10_dec,
        z_scaled,
        batch=z_scaled.shape[0],
        dtype=dec_dtype,
        enable_grad=True,
        output_cpu=False,
    )


def latent_proxy_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: torch.Tensor,
    *,
    is_identity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth-L1 + Haar DWT high-frequency brushstroke band loss."""
    ps = F.smooth_l1_loss(pred.float(), target.float(), beta=0.06, reduction="none").mean(dim=(1, 2, 3))
    w = (0.30 + 0.70 * lam.float().clamp(0.0, 1.0)) if is_identity is None else torch.where(
        is_identity, torch.ones_like(lam), 0.30 + 0.70 * lam.float().clamp(0.0, 1.0)
    )
    fp, ft = dwt2_haar(pred.float()), dwt2_haar(target.float())
    bands = sum(bw * F.l1_loss(a, b) for bw, a, b in zip((1.0, 1.0, 0.5), fp[1:], ft[1:]))
    return (ps * w).mean() + 0.25 * bands


def _gram_cos_loss(fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    """Scale-invariant normalized Gram matrix cosine distance capturing color-stroke covariance."""
    B, C, H, W = fa.shape
    xa = fa.float().view(B, C, H * W)
    xb = fb.float().view(B, C, H * W)
    ga = torch.bmm(xa, xa.transpose(1, 2)).flatten(1) / float(C * H * W)
    gb = torch.bmm(xb, xb.transpose(1, 2)).flatten(1) / float(C * H * W)
    return (1.0 - (F.normalize(ga, dim=-1) * F.normalize(gb, dim=-1)).sum(dim=-1)).mean() + 0.5 * F.l1_loss(ga, gb)


def _patch_swd(fa: torch.Tensor, fb: torch.Tensor, *, num_proj: int = 64) -> torch.Tensor:
    """Spatial-agnostic Sliced Wasserstein Distance over multi-scale VGG feature vectors."""
    B, C, H, W = fa.shape
    a = fa.float().view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
    b = fb.float().view(B, C, H * W).transpose(1, 2)
    dirs = torch.randn(C, num_proj, device=fa.device, dtype=torch.float32)
    dirs = F.normalize(dirs, dim=0)
    pa, _ = torch.sort(a @ dirs, dim=1)
    pb, _ = torch.sort(b @ dirs, dim=1)
    return F.l1_loss(pa, pb)


def oklab_moment_loss(rgb_pred: torch.Tensor, rgb_ref: torch.Tensor) -> torch.Tensor:
    """Oklab global tone & chroma mean/std matching."""
    lp = rgb_to_oklab(rgb_pred.float().clamp(0.0, 1.0))
    lr = rgb_to_oklab(rgb_ref.float().clamp(0.0, 1.0))
    mu_p, sd_p = lp.mean(dim=(2, 3)), lp.std(dim=(2, 3), unbiased=False)
    mu_r, sd_r = lr.mean(dim=(2, 3)), lr.std(dim=(2, 3), unbiased=False)
    return F.l1_loss(mu_p, mu_r) + F.l1_loss(sd_p, sd_r)


def gray_sobel_content_loss(rgb_pred: torch.Tensor, rgb_src: torch.Tensor) -> torch.Tensor:
    """Mild 64x64 luminance edge orientation preservation (allows stroke re-texturing inside objects)."""
    gp = F.interpolate(rgb_to_gray(rgb_pred.float()), size=(64, 64), mode="area")
    gs = F.interpolate(rgb_to_gray(rgb_src.float()), size=(64, 64), mode="area")
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gp.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    gx_p, gy_p = F.conv2d(gp, kx, padding=1), F.conv2d(gp, ky, padding=1)
    gx_s, gy_s = F.conv2d(gs, kx, padding=1), F.conv2d(gs, ky, padding=1)
    return 0.5 * (F.l1_loss(gx_p, gx_s) + F.l1_loss(gy_p, gy_s))


class ECorePerceptualCritic(nn.Module):
    """Frozen VGG-16 Gram + Mean/Std + Patch-SWD + Oklab + Grayscale Structure critic."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.vgg = FrozenVGGStyleCritic(device=device)

    def _vgg_feats(self, rgb01: torch.Tensor, size: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = rgb01.float().clamp(0.0, 1.0)
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - self.vgg.im_mean) / self.vgg.im_std
        with bf16_autocast(str(x.device)):
            f1 = self.vgg.s1(x)
            f2 = self.vgg.s2(f1)
            f3 = self.vgg.s3(f2)
            f4 = self.vgg.s4_3(self.vgg.s4_2(f3))
        return f1, f2, f3, f4

    def _style_vec_from_feats(self, feats: tuple[torch.Tensor, ...]) -> torch.Tensor:
        parts = [
            F.normalize(self.vgg._feat_moments(f.float()), dim=-1)
            for f in feats
        ]
        return torch.cat(parts, dim=-1) / 2.0

    def style_signature(self, rgb01: torch.Tensor) -> torch.Tensor:
        return self._style_vec_from_feats(self._vgg_feats(rgb01, size=256))

    def forward_swap_pair(
        self,
        s10_dec: nn.Module,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        x_src: torch.Tensor,
        x_ref_a: torch.Tensor,
        x_ref_b: torch.Tensor,
        lam: torch.Tensor,
        *,
        margin: float = 0.08,
        use_ckpt: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B = z_a.shape[0]
        z_ab = torch.cat([z_a, z_b], dim=0)
        x_ab = cp.checkpoint(lambda z: _decode_s10_diff(s10_dec, z), z_ab, use_reentrant=False) if use_ckpt else _decode_s10_diff(s10_dec, z_ab)
        xa, xb = x_ab[:B], x_ab[B:]

        with torch.no_grad():
            fr_all = self._vgg_feats(torch.cat([x_ref_a, x_ref_b], dim=0))
            fa_r = tuple(f[:B] for f in fr_all)
            fb_r = tuple(f[B:] for f in fr_all)
            va_r = self._style_vec_from_feats(fa_r)
            vb_r = self._style_vec_from_feats(fb_r)
            cs_gt = self.vgg.gray_content_vec(x_src)

        fp_all = cp.checkpoint(self._vgg_feats, x_ab, use_reentrant=False) if use_ckpt else self._vgg_feats(x_ab)
        fa_p = tuple(f[:B] for f in fp_all)
        fb_p = tuple(f[B:] for f in fp_all)
        va_p = self._style_vec_from_feats(fa_p)
        vb_p = self._style_vec_from_feats(fb_p)
        c_ab_p = cp.checkpoint(self.vgg.gray_content_vec, x_ab, use_reentrant=False) if use_ckpt else self.vgg.gray_content_vec(x_ab)
        ca_p, cb_p = c_ab_p[:B], c_ab_p[B:]

        # 1. Contrastive swap ranking + direct cosine on multi-scale VGG moments
        d_aa = 1.0 - (va_p * va_r).sum(dim=-1)
        d_ab = 1.0 - (va_p * vb_r).sum(dim=-1)
        d_bb = 1.0 - (vb_p * vb_r).sum(dim=-1)
        d_ba = 1.0 - (vb_p * va_r).sum(dim=-1)
        l_rank = 0.5 * (F.relu(margin + d_aa - d_ab).mean() + F.relu(margin + d_bb - d_ba).mean())
        l_dir = 0.5 * (d_aa.mean() + d_bb.mean())

        # 2. Multi-scale Gram covariance + Patch-SWD (spatial-agnostic brushstroke texture)
        w_lam = lam.float().clamp(0.2, 1.0)
        l_gram = 0.5 * (
            sum(gw * _gram_cos_loss(fp, fr) for fp, fr, gw in zip(fa_p, fa_r, (0.6, 1.0, 1.2, 1.0)))
            + sum(gw * _gram_cos_loss(fp, fr) for fp, fr, gw in zip(fb_p, fb_r, (0.6, 1.0, 1.2, 1.0)))
        ) * 0.25

        l_swd = 0.25 * (
            _patch_swd(fa_p[1], fa_r[1], num_proj=48) + _patch_swd(fa_p[2], fa_r[2], num_proj=48)
            + _patch_swd(fb_p[1], fb_r[1], num_proj=48) + _patch_swd(fb_p[2], fb_r[2], num_proj=48)
        )

        l_ok = 0.5 * (oklab_moment_loss(xa, x_ref_a) + oklab_moment_loss(xb, x_ref_b))
        l_cont = 0.5 * (F.l1_loss(ca_p, cs_gt) + F.l1_loss(cb_p, cs_gt))
        l_edge = 0.5 * (gray_sobel_content_loss(xa, x_src) + gray_sobel_content_loss(xb, x_src))

        # Keep output latent amplitude aligned with SD1.5 P-Core manifold (~0.80 std)
        l_lat_std = 0.5 * (
            (z_a.float().flatten(1).std(1) - 0.80).square().mean()
            + (z_b.float().flatten(1).std(1) - 0.80).square().mean()
        )

        s_scale = w_lam.mean()
        loss = (
            1.15 * s_scale * (l_rank + 0.45 * l_dir)
            + 0.70 * s_scale * l_gram
            + 0.30 * s_scale * l_swd
            + 0.35 * s_scale * l_ok
            + 0.42 * l_cont
            + 0.15 * l_edge
            + 0.12 * l_lat_std
        )
        follow = 0.5 * ((d_aa < d_ab).float().mean().item() + (d_bb < d_ba).float().mean().item())
        return loss, {
            "l_swap": float(loss.item()),
            "l_gram": float(l_gram.item()),
            "l_swd": float(l_swd.item()),
            "l_cont": float(l_cont.item()),
            "swap_follow": follow,
        }

    def forward_proxy_pixel(
        self,
        s10_dec: nn.Module,
        z_pred: torch.Tensor,
        x_gt: torch.Tensor,
        x_ref: torch.Tensor,
        *,
        use_ckpt: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        xp = cp.checkpoint(lambda z: _decode_s10_diff(s10_dec, z), z_pred, use_reentrant=False) if use_ckpt else _decode_s10_diff(s10_dec, z_pred)
        with torch.no_grad():
            fr = self._vgg_feats(x_ref)
            vr = self._style_vec_from_feats(fr)
            cg = self.vgg.gray_content_vec(x_gt)
        fp = cp.checkpoint(self._vgg_feats, xp, use_reentrant=False) if use_ckpt else self._vgg_feats(xp)
        vp = self._style_vec_from_feats(fp)
        cp_v = cp.checkpoint(self.vgg.gray_content_vec, xp, use_reentrant=False) if use_ckpt else self.vgg.gray_content_vec(xp)

        l_dir = (1.0 - (vp * vr).sum(dim=-1)).mean()
        l_gram = 0.25 * sum(gw * _gram_cos_loss(a, b) for a, b, gw in zip(fp, fr, (0.6, 1.0, 1.2, 1.0)))
        l_swd = 0.5 * (_patch_swd(fp[1], fr[1], num_proj=48) + _patch_swd(fp[2], fr[2], num_proj=48))
        l_ok = oklab_moment_loss(xp, x_gt)
        l_cont = F.l1_loss(cp_v, cg)
        loss = 0.45 * l_dir + 0.60 * l_gram + 0.30 * l_swd + 0.25 * l_ok + 0.35 * l_cont
        return loss, {"l_px_gram": float(l_gram.item()), "l_px_cont": float(l_cont.item())}
