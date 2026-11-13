"""WEAVE V6 Models:
1. C3UNet (~2.8M params): Multi-scale Conv U-Net for both:
   - D1 Direct Operator: z_hat = G_phi(c, s)
   - F1 Content-Canonicalized Conditional Flow (C³-Flow): v_theta(z_tau, tau, c, s)
   Features:
   - Explicit multi-scale canonical content injection at 64x64, 32x32, 16x16.
   - InstanceNorm2d(affine=False) + per-layer Style-ID & tau AdaLN/FiLM + style-conditioned
     channel mixing so Style-ID directly controls every spatial scale.
   - Full 4x64x64 output (including HH subband), zero-initialized residual head.
2. ConditionalScoreNet (~2.6M params): Style-conditional multi-scale denoiser
   eps_theta(z_sigma, sigma, s) for the V6 backup Score Difference route:
   g_{s -> t} = (eps_theta(z_sigma, sigma, s) - eps_theta(z_sigma, sigma, t)) / sigma.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous scalar embedding for tau or log(sigma)."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = t.float().view(-1, 1) * freqs.view(1, -1) * 2.0 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class StyleAdaLNBlock(nn.Module):
    """Conv block with InstanceNorm + (Style-ID, tau) AdaLN + Content injection."""

    def __init__(self, channels: int, cond_dim: int, content_channels: int = 0):
        super().__init__()
        self.has_content = content_channels > 0
        if self.has_content:
            self.content_norm = nn.InstanceNorm2d(content_channels, affine=False, eps=1e-5)
            self.content_proj = nn.Conv2d(content_channels, channels, 1)

        self.norm1 = nn.InstanceNorm2d(channels, affine=False, eps=1e-5)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=False, eps=1e-5)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        # Predicts (gamma1, beta1, gamma2, beta2, gate) from (style_id, tau)
        self.adaln = nn.Linear(cond_dim, channels * 5)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        content_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        if self.has_content and content_feat is not None:
            c_norm = self.content_norm(content_feat)
            h = h + self.content_proj(c_norm)

        g1, b1, g2, b2, gate = self.adaln(cond).chunk(5, dim=-1)
        g1 = g1[:, :, None, None]
        b1 = b1[:, :, None, None]
        g2 = g2[:, :, None, None]
        b2 = b2[:, :, None, None]
        gate = gate[:, :, None, None]

        u = self.norm1(h) * (1.0 + g1) + b1
        u = self.conv1(F.silu(u))
        u = self.norm2(u) * (1.0 + g2) + b2
        u = self.conv2(F.silu(u))
        return x + u * (1.0 + gate)


class StyleChannelPaletteHead(nn.Module):
    """Per-style learnable subband affine + 4x4 channel covariance mixer on DWT bands."""

    def __init__(self, num_styles: int = 5):
        super().__init__()
        # Per-style 4x4 channel mixing matrix (initialized to 0 residual) and bias per band
        self.mix_ll = nn.Parameter(torch.zeros(num_styles, 4, 4))
        self.bias_ll = nn.Parameter(torch.zeros(num_styles, 4, 1, 1))
        self.scale_hf = nn.Parameter(torch.zeros(num_styles, 12, 1, 1))
        self.bias_hf = nn.Parameter(torch.zeros(num_styles, 12, 1, 1))

    def forward(self, z_base: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = dwt2_haar(z_base.float())
        # match the parameter dtype: dwt2_haar always returns fp32, which breaks
        # half-precision deployment ("expected scalar type Float but found Half").
        _dt = self.mix_ll.dtype
        ll, lh, hl, hh = ll.to(_dt), lh.to(_dt), hl.to(_dt), hh.to(_dt)
        B, C, H, W = ll.shape
        sid = style_id.long()
        w_ll = self.mix_ll[sid]  # (B, 4, 4)
        b_ll = self.bias_ll[sid]  # (B, 4, 1, 1)

        ll_flat = ll.reshape(B, C, H * W)
        ll_delta = torch.bmm(w_ll, ll_flat).reshape(B, C, H, W) + b_ll

        hf = torch.cat([lh, hl, hh], dim=1)
        s_hf = self.scale_hf[sid]
        b_hf = self.bias_hf[sid]
        hf_delta = hf * s_hf + b_hf
        d_lh, d_hl, d_hh = hf_delta.chunk(3, dim=1)

        return idwt2_haar(ll_delta, d_lh, d_hl, d_hh).to(dtype=z_base.dtype)


class StyleKeyResidual(nn.Module):
    """Plan S6: per-style learnable prototype keys with low-rank residual attention.

    Category-conditioned PROTOTYPES, not attention over arbitrary style-reference images.
    Each style owns K learnable keys of rank r; the spatial features at the 16x16
    bottleneck query them, so the model can select a target-style prototype BY CONTENT
    POSITION instead of receiving one global Style-ID vector.

    The output projection is zero-initialised, so adding this module does not change the
    initial function at all -- the mechanism has to earn its effect during training.
    """

    def __init__(self, channels: int, num_styles: int = 5, n_keys: int = 8,
                 rank: int = 32):
        super().__init__()
        self.n_keys = n_keys
        self.rank = rank
        self.keys = nn.Parameter(torch.randn(num_styles, n_keys, rank) * 0.02)
        self.q = nn.Conv2d(channels, rank, 1)
        self.out = nn.Conv2d(rank, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.scale = float(rank) ** -0.5

    def forward(self, h: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        q = self.q(h).flatten(2).transpose(1, 2)                  # (B, HW, r)
        k = self.keys[style_id.long()]                            # (B, K, r)
        att = torch.softmax(q @ k.transpose(1, 2) * self.scale, dim=-1)   # (B, HW, K)
        o = (att @ k).transpose(1, 2).reshape(B, self.rank, H, W)
        return h + self.out(o)


class C3UNet(nn.Module):
    """Multi-scale Conv U-Net (~2.9M params) for D1 (Direct) and F1 (C³-Flow).

    Scales:
      - Scale 1: 64x64 (c1 = base_dim)
      - Scale 2: 32x32 (c2 = base_dim * 2, fused with exact Haar DWT bands of z and c)
      - Scale 3: 16x16 (c3 = base_dim * 2)
    """

    def __init__(
        self,
        base_dim: int = 96,
        num_styles: int = 5,
        mode: str = "flow",
        no_content: bool = False,
        no_style: bool = False,
        content_mode: str = "full",
        inject_64: bool = True,
        inject_32: bool = True,
        inject_16: bool = True,
        folded: bool = False,
        style_keys: bool = False,
        style_keys_n: int = 8,
        style_keys_rank: int = 32,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.num_styles = num_styles
        self.mode = mode  # "direct" or "flow"
        # --- V6 ablation switches (docs/v6/V6_ABLATION_INVENTORY.md) -------------
        # content_mode:
        #   "full"      normal
        #   "no_inject" zero the per-block content injection, but the stem still sees
        #               z_in (which equals c in direct mode), so content is NOT removed.
        #               This isolates the cheap 0.6%-of-params injection path.
        #   "none"      zero BOTH z_in and c, so the network sees no content at all and
        #               the direct residual starts from 0. This is the true
        #               style-only-generation control.
        # no_style: pin Style-ID to 0 (same weights/params).
        self.content_mode = "no_inject" if no_content and content_mode == "full" \
            else content_mode
        self.no_content = self.content_mode != "full"
        self.no_style = bool(no_style)
        # Per-scale content injection switches (plan S1/S2/S3). Passing None to a block
        # exactly disables its content_proj branch, leaving every other weight untouched.
        self.inject_64 = bool(inject_64)
        self.inject_32 = bool(inject_32)
        self.inject_16 = bool(inject_16)
        # `folded` is the zero-training EXACT deployment compression for direct mode
        # (plan section 4): the 12-channel stem becomes 4-channel because
        # [z_in, c, z_in-c] = [c, c, 0] when z_in == c, and the constant tau embedding is
        # folded into the AdaLN biases. Only valid for mode="direct"; the conversion is
        # done by tools/v6_24h/fold_direct.py and verified output-for-output.
        self.folded = bool(folded)
        if self.folded and mode != "direct":
            raise ValueError("folded=True is only valid for mode='direct'")
        # plan S6: per-style prototype keys at the 16x16 bottleneck
        self.style_keys = bool(style_keys)

        c1, c2, c3 = base_dim, base_dim * 2, base_dim * 2
        cond_dim = base_dim * 2

        self.style_emb = nn.Embedding(num_styles, cond_dim)
        self.style_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input stem at 64x64: takes [z_tau, c, z_tau - c] (12 channels)
        self.stem64 = nn.Conv2d(4 if self.folded else 12, c1, 3, padding=1)

        # Encoder 64x64
        self.enc64_1 = StyleAdaLNBlock(c1, cond_dim, content_channels=4)
        self.enc64_2 = StyleAdaLNBlock(c1, cond_dim, content_channels=4)
        self.down32 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)

        # At 32x32, inject 16-ch Haar DWT subbands of c
        self.enc32_1 = StyleAdaLNBlock(c2, cond_dim, content_channels=16)
        self.enc32_2 = StyleAdaLNBlock(c2, cond_dim, content_channels=16)
        self.down16 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)

        # Bottleneck at 16x16 with pooled 16-ch DWT content features
        self.mid16_1 = StyleAdaLNBlock(c3, cond_dim, content_channels=16)
        self.mid16_2 = StyleAdaLNBlock(c3, cond_dim, content_channels=16)
        self.skr = (StyleKeyResidual(c3, num_styles=num_styles, n_keys=style_keys_n,
                                     rank=style_keys_rank) if self.style_keys else None)

        # Decoder 32x32
        self.up32 = nn.Conv2d(c3, c2, 3, padding=1)
        self.dec32_proj = nn.Conv2d(c2 * 2, c2, 1)
        self.dec32_1 = StyleAdaLNBlock(c2, cond_dim, content_channels=16)
        self.dec32_2 = StyleAdaLNBlock(c2, cond_dim, content_channels=16)

        # Decoder 64x64
        self.up64 = nn.Conv2d(c2, c1, 3, padding=1)
        self.dec64_proj = nn.Conv2d(c1 * 2, c1, 1)
        self.dec64_1 = StyleAdaLNBlock(c1, cond_dim, content_channels=4)
        self.dec64_2 = StyleAdaLNBlock(c1, cond_dim, content_channels=4)

        # Final style-conditioned output head (4x64x64 full latent + 16-ch DWT subband head)
        self.out_norm = nn.InstanceNorm2d(c1, affine=False, eps=1e-5)
        self.out_adaln = nn.Linear(cond_dim, c1 * 2)
        self.out_conv64 = nn.Conv2d(c1, 4, 3, padding=1)
        self.out_dwt32 = nn.Conv2d(c2, 16, 3, padding=1)
        self.palette_head = StyleChannelPaletteHead(num_styles=num_styles)

        nn.init.zeros_(self.out_adaln.weight)
        nn.init.zeros_(self.out_adaln.bias)
        nn.init.zeros_(self.out_conv64.weight)
        nn.init.zeros_(self.out_conv64.bias)
        nn.init.zeros_(self.out_dwt32.weight)
        nn.init.zeros_(self.out_dwt32.bias)

    def forward(
        self,
        z_in: torch.Tensor,
        tau: torch.Tensor | float,
        c: torch.Tensor,
        style_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        If mode == 'direct': returns reconstructed latent z_hat = c + delta(c, s).
        If mode == 'flow':   returns velocity v_theta(z_tau, tau, c, s).
        """
        B = z_in.shape[0]
        if self.content_mode == "none":
            z_in = torch.zeros_like(z_in)
            c = torch.zeros_like(c)
        elif self.content_mode == "no_inject":
            c = torch.zeros_like(c)
        if self.no_style:
            style_id = torch.zeros_like(style_id)
        if not torch.is_tensor(tau):
            tau_t = torch.full((B,), float(tau), device=z_in.device, dtype=torch.float32)
        else:
            tau_t = tau.float().view(B)

        s_emb = self.style_mlp(self.style_emb(style_id.long()))
        if self.folded:
            # tau is a constant 0 in direct mode; its contribution was folded into the
            # AdaLN biases at conversion time, so no time network is needed here.
            cond = s_emb
        else:
            # match the embedding dtype: sinusoidal_embedding always returns fp32, which
            # breaks half-precision deployment ("mat1 Float vs mat2 Half").
            t_emb = self.time_mlp(
                sinusoidal_embedding(tau_t, self.base_dim).to(s_emb.dtype))
            cond = s_emb + t_emb

        # Content multi-scale pyramid: 64x64 raw c, 32x32 Haar DWT(c), 16x16 pooled DWT(c)
        c_ll, c_lh, c_hl, c_hh = dwt2_haar(c.float())
        c32 = torch.cat([c_ll, c_lh, c_hl, c_hh], dim=1).to(dtype=z_in.dtype)
        c16 = F.avg_pool2d(c32, kernel_size=2, stride=2)

        x0 = torch.cat([z_in, c, z_in - c], dim=1)
        if self.folded:
            # exact: in direct mode z_in == c, so the 12-channel [z_in, c, z_in-c] has
            # 4 zero channels and 8 duplicated ones -> equivalent to feeding z_in alone.
            x0 = z_in
        h64 = self.stem64(x0)
        # Per-scale injection gates (plan S1/S2/S3): None disables only that scale's
        # content_proj branch; all other weights are untouched.
        i64 = c if self.inject_64 else None
        i32 = c32 if self.inject_32 else None
        i16 = c16 if self.inject_16 else None
        h64 = self.enc64_1(h64, cond, i64)
        h64 = self.enc64_2(h64, cond, i64)

        h32 = self.down32(h64)
        h32 = self.enc32_1(h32, cond, i32)
        h32 = self.enc32_2(h32, cond, i32)

        h16 = self.down16(h32)
        h16 = self.mid16_1(h16, cond, i16)
        h16 = self.mid16_2(h16, cond, i16)
        if self.skr is not None:
            h16 = self.skr(h16, style_id)

        u32 = F.interpolate(h16, scale_factor=2.0, mode="nearest")
        u32 = self.up32(u32)
        u32 = self.dec32_proj(torch.cat([u32, h32], dim=1))
        u32 = self.dec32_1(u32, cond, i32)
        u32 = self.dec32_2(u32, cond, i32)

        u64 = F.interpolate(u32, scale_factor=2.0, mode="nearest")
        u64 = self.up64(u64)
        u64 = self.dec64_proj(torch.cat([u64, h64], dim=1))
        u64 = self.dec64_1(u64, cond, i64)
        u64 = self.dec64_2(u64, cond, i64)

        go, bo = self.out_adaln(cond).chunk(2, dim=-1)
        u_out = F.silu(self.out_norm(u64) * (1.0 + go[:, :, None, None]) + bo[:, :, None, None])
        delta_spatial = self.out_conv64(u_out)

        # Combine 32x32 DWT subband head (including HH) + 64x64 spatial head + palette head
        dwt_raw = self.out_dwt32(F.silu(u32)).float()
        d_ll, d_lh, d_hl, d_hh = dwt_raw.chunk(4, dim=1)
        delta_dwt = idwt2_haar(d_ll, d_lh, d_hl, d_hh).to(dtype=z_in.dtype)
        delta_pal = self.palette_head(c, style_id)

        total_delta = delta_spatial + delta_dwt + delta_pal
        if self.mode == "direct":
            return c + total_delta
        return total_delta


class ConditionalScoreNet(nn.Module):
    """Multi-scale style-conditional denoiser eps_theta(z_sigma, sigma, s) for Score Difference.

    Supports any spatial resolution divisible by 4 (e.g. 16x16, 32x32 crops or 64x64 full latents).
    """

    def __init__(self, base_dim: int = 96, num_styles: int = 5):
        super().__init__()
        self.base_dim = base_dim
        c1, c2, c3 = base_dim, base_dim * 2, base_dim * 2
        cond_dim = base_dim * 2

        self.style_emb = nn.Embedding(num_styles, cond_dim)
        self.style_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.sigma_mlp = nn.Sequential(
            nn.Linear(base_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.stem = nn.Conv2d(4, c1, 3, padding=1)
        self.enc1_1 = StyleAdaLNBlock(c1, cond_dim, content_channels=0)
        self.enc1_2 = StyleAdaLNBlock(c1, cond_dim, content_channels=0)
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)

        self.enc2_1 = StyleAdaLNBlock(c2, cond_dim, content_channels=0)
        self.enc2_2 = StyleAdaLNBlock(c2, cond_dim, content_channels=0)
        self.down4 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)

        self.mid_1 = StyleAdaLNBlock(c3, cond_dim, content_channels=0)
        self.mid_2 = StyleAdaLNBlock(c3, cond_dim, content_channels=0)

        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.dec2_proj = nn.Conv2d(c2 * 2, c2, 1)
        self.dec2_1 = StyleAdaLNBlock(c2, cond_dim, content_channels=0)
        self.dec2_2 = StyleAdaLNBlock(c2, cond_dim, content_channels=0)

        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.dec1_proj = nn.Conv2d(c1 * 2, c1, 1)
        self.dec1_1 = StyleAdaLNBlock(c1, cond_dim, content_channels=0)
        self.dec1_2 = StyleAdaLNBlock(c1, cond_dim, content_channels=0)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, 4, 3, padding=1)

    def forward(
        self,
        z_sigma: torch.Tensor,
        sigma: torch.Tensor | float,
        style_id: torch.Tensor,
    ) -> torch.Tensor:
        B = z_sigma.shape[0]
        if not torch.is_tensor(sigma):
            sig_t = torch.full((B,), float(sigma), device=z_sigma.device, dtype=torch.float32)
        else:
            sig_t = sigma.float().view(B)

        log_sig = torch.log(sig_t.clamp(min=1e-4))
        cond = self.style_mlp(self.style_emb(style_id.long())) + self.sigma_mlp(
            sinusoidal_embedding(log_sig, self.base_dim)
        )

        h1 = self.stem(z_sigma)
        h1 = self.enc1_1(h1, cond)
        h1 = self.enc1_2(h1, cond)

        h2 = self.down2(h1)
        h2 = self.enc2_1(h2, cond)
        h2 = self.enc2_2(h2, cond)

        h4 = self.down4(h2)
        h4 = self.mid_1(h4, cond)
        h4 = self.mid_2(h4, cond)

        u2 = self.up2(F.interpolate(h4, scale_factor=2.0, mode="nearest"))
        u2 = self.dec2_proj(torch.cat([u2, h2], dim=1))
        u2 = self.dec2_1(u2, cond)
        u2 = self.dec2_2(u2, cond)

        u1 = self.up1(F.interpolate(u2, scale_factor=2.0, mode="nearest"))
        u1 = self.dec1_proj(torch.cat([u1, h1], dim=1))
        u1 = self.dec1_1(u1, cond)
        u1 = self.dec1_2(u1, cond)

        return self.out_conv(F.silu(self.out_norm(u1)))
