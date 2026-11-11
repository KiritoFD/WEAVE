"""Losses for SD0-S10 V2 (C0 reconstruction + C1 response distillation).

All wavelet math is FP32. The student decoder D_theta contains EXACTLY ONE
post_quant_conv (handled by StudentDecoder in decoder_prune.py), so callers pass
the RAW unscaled latent z and the wrapper applies pqc internally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# --------------------------------------------------------------------- wavelets
def haar_dwt(x: torch.Tensor, levels: int = 2):
    """2-D Haar DWT. Returns list of (LH, HL, HH, LL) per level, finest first.

    x: (B,3,H,W) FP32. Each band is (B,3,H/2^l,W/2^l).
    Coefficients use the orthonormal-ish 0.5 scaling: a+b+c+d is the average,
    the three differences are *0.5 so magnitudes stay comparable to pixel L1.
    """
    x = x.float()
    out = []
    cur = x
    for _ in range(levels):
        a = cur[:, :, 0::2, 0::2]
        b = cur[:, :, 1::2, 0::2]
        c = cur[:, :, 0::2, 1::2]
        d = cur[:, :, 1::2, 1::2]
        lh = (a - b + c - d) * 0.5
        hl = (a + b - c - d) * 0.5
        hh = (a - b - c + d) * 0.5
        ll = (a + b + c + d) * 0.5
        out.append((lh, hl, hh, ll))
        cur = ll
    return out


def estimate_band_scales(loader_pairs, device="cuda", n_batches=64):
    """FIXED per-band L1 scale estimated from the TRAINING set (GT only).

    Called ONCE before training; never recomputed from model outputs, otherwise
    the loss would be gameable by shrinking the whole representation.
    Returns dict {(level, band_idx): float}.
    """
    acc = defaultdict(float)
    cnt = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader_pairs):
            if i >= n_batches:
                break
            x = x.to(device).float()
            for lvl, (lh, hl, hh, _) in enumerate(haar_dwt(x, 2)):
                for bi, band in enumerate((lh, hl, hh)):
                    acc[(lvl, bi)] += float(band.abs().mean())
            cnt += 1
    return {k: max(v / max(cnt, 1), EPS) for k, v in acc.items()}


class WaveletHF(nn.Module):
    """逐系数 Haar 高频监督 + 弱能量正则.

    L_hf  = sum_{l,b} mean |W_lb(y) - W_lb(x)| / a_lb     (a_lb FIXED)
    L_eng = sum_{l,b} Huber( log((std(W_lb(y))+eps)/(std(W_lb(x))+eps)) )
    The energy term is OFF by default (weight 0) and may only be enabled after
    a measured systematic amplitude deficit. It constrains MAGNITUDE only; the
    per-coefficient term is what enforces spatial/phase alignment.
    """

    def __init__(self, band_scales, levels=2):
        super().__init__()
        self.levels = levels
        self.register_buffer("a", torch.ones(levels, 3))
        for (lvl, bi), v in band_scales.items():
            if lvl < levels and bi < 3:
                self.a[lvl, bi] = float(v)

    def forward(self, y, x):
        Wy, Wx = haar_dwt(y.float(), self.levels), haar_dwt(x.float(), self.levels)
        l_hf = y.new_zeros(())
        l_eng = y.new_zeros(())
        per_band = {}
        for lvl in range(self.levels):
            for bi in range(3):
                cy, cx = Wy[lvl][bi], Wx[lvl][bi]
                a = self.a[lvl, bi].clamp_min(EPS)
                err = (cy - cx).abs().mean() / a
                l_hf = l_hf + err
                sy = cy.flatten(1).std(dim=1).mean()
                sx = cx.flatten(1).std(dim=1).mean()
                l_eng = l_eng + F.huber_loss(
                    torch.log((sy + EPS) / (sx + EPS)),
                    torch.zeros_like(sy), delta=0.5)
                per_band[f"L{lvl}_b{bi}"] = float(err.detach())
        return l_hf, l_eng, per_band


# --------------------------------------------------------------------- response
def make_perturbation(z: torch.Tensor, kind: str, frac: float,
                      z_partner: torch.Tensor | None = None,
                      generator: torch.Generator | None = None):
    """δ with RMS(δ) = frac * RMS(z), sampled per-sample.

    kinds:
      corr_gauss : multiscale correlated Gaussian field (blur of white noise)
      interp     : small step toward another REAL latent (z_partner)
      posterior  : posterior-style local perturbation (smooth + channel-wise)
    """
    z = z.float()
    B, C = z.shape[0], z.shape[1]
    # `posterior` builds a (B,4,1,1) channel field; on a non-4-channel input it
    # would silently broadcast to the wrong shape. The production latent is
    # always 4-channel, so assert it explicitly rather than corrupt the loss.
    assert C == 4, f"make_perturbation expects 4 latent channels, got {C}"
    dev, dt = z.device, z.dtype
    if generator is None:
        noise = torch.randn(z.shape, device=dev, dtype=dt)
    else:
        noise = torch.randn(z.shape, device=dev, dtype=dt, generator=generator)

    if kind == "corr_gauss":
        # multi-scale: mix white noise with 2x and 4x upsampled smooth fields
        small = torch.randn(B, 4, 16, 16, device=dev, dtype=dt,
                            generator=generator)
        mid = F.interpolate(small, size=(64, 64), mode="bilinear",
                            align_corners=False)
        tiny = torch.randn(B, 4, 8, 8, device=dev, dtype=dt,
                           generator=generator)
        coarse = F.interpolate(tiny, size=(64, 64), mode="bilinear",
                               align_corners=False)
        delta = 0.5 * noise + 0.3 * mid + 0.2 * coarse
    elif kind == "interp":
        assert z_partner is not None
        delta = z_partner.float() - z
    elif kind == "posterior":
        chan = torch.randn(B, 4, 1, 1, device=dev, dtype=dt,
                           generator=generator)
        delta = 0.6 * noise + 0.4 * chan
    else:
        raise ValueError(kind)

    # normalise so RMS(delta) == frac * RMS(z), per sample
    zn = z.flatten(1).norm(dim=1).view(-1, 1, 1, 1) / (z[0].numel() ** 0.5)
    dn = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1) / (delta[0].numel() ** 0.5)
    return delta / (dn + EPS) * (frac * zn)


def response_loss(student, teacher, z, delta, a_delta_floor=0.02):
    """Finite-difference Secant/Jacobian response matching.

    Δy_S = D_S(z+δ) - D_S(z),  Δy_T = D_T(z+δ) - D_T(z)   (teacher detached)
    L = SmoothL1( (Δy_S - Δy_T) / (a_δ + eps) ),  a_δ from teacher response.
    Only FIRST-order autograd: no full Jacobian, no second derivatives.
    """
    with torch.no_grad():
        y0_t = teacher(z).float().detach()
        y1_t = teacher(z + delta).float().detach()
        dy_t = y1_t - y0_t
        a_delta = dy_t.abs().mean().clamp_min(a_delta_floor).detach()
    y0_s = student(z).float()
    y1_s = student(z + delta).float()
    dy_s = y1_s - y0_s
    loss = F.smooth_l1_loss(dy_s / (a_delta + EPS), dy_t / (a_delta + EPS))
    return loss, {
        "resp_amp_ratio": float((dy_s.abs().mean() /
                                 (dy_t.abs().mean() + EPS)).detach()),
        "resp_cos": float(F.cosine_similarity(
            dy_s.flatten(1), dy_t.flatten(1), dim=1).mean().detach()),
        "a_delta": float(a_delta.detach()),
    }


def off_manifold_loss(student, teacher, z_pert, weight_scale=1.0):
    """Weak teacher consistency on perturbed (GT-less) latents."""
    with torch.no_grad():
        y_t = teacher(z_pert).float().detach()
    y_s = student(z_pert).float()
    return F.l1_loss(y_s, y_t), {"off_l1": float((y_s - y_t).abs().mean().detach())}


# --------------------------------------------------------------------- recon
class ReconLoss(nn.Module):
    """C0: Charbonnier + λ_p LPIPS@512 (+ optional HF/energy handled outside)."""

    def __init__(self, lpips_fn=None, lam_p=0.5, lpips_size=512):
        super().__init__()
        self.lpips_fn = lpips_fn
        self.lam_p = lam_p
        self.lpips_size = lpips_size

    def forward(self, x, y):
        y = y.float()
        d = y - x
        l1 = torch.sqrt(d * d + 1e-6).mean()          # Charbonnier
        parts = {"charb": float(l1.detach()), "l1": float(d.abs().mean().detach())}
        total = l1
        if self.lpips_fn is not None:
            if self.lpips_size and self.lpips_size != y.shape[-1]:
                ys = F.interpolate(y, self.lpips_size, mode="bilinear",
                                   align_corners=False)
                xs = F.interpolate(x, self.lpips_size, mode="bilinear",
                                   align_corners=False)
            else:
                ys, xs = y, x
            lp = self.lpips_fn(ys.clamp(-1, 1), xs.clamp(-1, 1)).mean()
            total = total + self.lam_p * lp
            parts["lpips"] = float(lp.detach())
        return total, parts


from collections import defaultdict  # noqa: E402  (used by estimate_band_scales)

def prune_with_indices(dec_src, widths, channel_index):
    """Cold-SD0: rebuild S10 using the SAME channel indices as the original S10.

    The original S10 was pruned from A0 with a Taylor importance order. Cold must
    reuse that exact index set (plan section 2.1), NOT a fresh random permutation.
    We copy weights along the saved indices from the OFFICIAL SD0 decoder.
    """
    from tools.vae_suite.decoder_prune import (A0_SRC, build_decoder, get_module,
                                               make_plan)
    tgt = {"A": widths[3], "B": widths[2], "C": widths[1], "D": widths[0]}
    idx = {k: torch.as_tensor(v, dtype=torch.long) for k, v in channel_index.items()}
    new = build_decoder(widths)
    with torch.no_grad():
        for path, kind, si, so in make_plan(dec_src):
            mo, mn = get_module(dec_src, path), get_module(new, path)
            io_i = idx[so] if (so is not None and
                               mo.weight.shape[0] == A0_SRC[so]) else None
            if kind == "conv":
                w = mo.weight
                ii = idx[si] if (si is not None and
                                 w.shape[1] == A0_SRC[si]) else None
                wn = w
                if io_i is not None:
                    wn = wn[io_i]
                if ii is not None:
                    wn = wn[:, ii]
                mn.weight.copy_(wn)
                if mo.bias is not None:
                    mn.bias.copy_(mo.bias if io_i is None else mo.bias[io_i])
            elif kind == "norm":
                w = mo.weight
                ii = idx[si] if (si is not None and
                                 w.numel() == A0_SRC[si]) else None
                mn.weight.copy_(w if ii is None else w[ii])
                mn.bias.copy_(mo.bias if ii is None else mo.bias[ii])
            elif kind == "linear":
                w = mo.weight
                ii = idx[si] if (si is not None and
                                 w.shape[1] == A0_SRC[si]) else None
                wn = w
                if io_i is not None:
                    wn = wn[io_i]
                if ii is not None:
                    wn = wn[:, ii]
                mn.weight.copy_(wn)
                if mo.bias is not None:
                    mn.bias.copy_(mo.bias if io_i is None else mo.bias[io_i])
    return new, {"widths_forward": widths, "stream_target": tgt,
                 "channel_index": channel_index,
                 "params": sum(q.numel() for q in new.parameters())}



