"""Pure-PyTorch Fixed-Shape Infra Optimizations for CLEAVE V35 (Zero External Engine).

Techniques:
  - BottleneckAttention SDPA replacement (exact weight equivalence, eliminates MHA fastpath overhead)
  - FastA0Encoder: drops unused logvar channels in quant_conv + fuses [-1, 1] scaling
  - FastConditioner: shared grayscale, pre-registered Sobel/Gauss/Laplacian buffers,
    ConvNeXt-S3 FP16, and pure-matmul Newton-Schulz Stiefel orthogonalization (CUDA-Graph safe)
  - HoistedPCoreStep: hoists step-invariant gamma blend, dino_proj, mid1/mid2 SPADE blocks,
    and cond_mlp static columns out of the 7-call ODE trajectory (14x fewer SPADE/dino_proj calls)
  - FixedShapeCompiledRuntime: per-batch torch.compile(mode="reduce-overhead") with persistent
    FX graph disk cache + automatic padding for tail batches.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18.ccf_model import BottleneckAttention, sinusoidal_time_emb


def enable_persistent_inductor_cache(cache_dir: Path | str) -> None:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(p))


def patch_bottleneck_sdpa(mod: nn.Module) -> int:
    """Replace nn.MultiheadAttention forward in BottleneckAttention with F.scaled_dot_product_attention."""
    n_patched = 0
    for m in mod.modules():
        if isinstance(m, BottleneckAttention) and not getattr(m, "_sdpa_patched", False):
            nh = m.attn.num_heads
            dim = m.attn.embed_dim
            hd = dim // nh
            w_i = m.attn.in_proj_weight
            b_i = m.attn.in_proj_bias
            w_o = m.attn.out_proj.weight
            b_o = m.attn.out_proj.bias

            def _make_fwd(norm, wi, bi, wo, bo, nheads, hdim):
                def _fwd(x: torch.Tensor) -> torch.Tensor:
                    B, C, H, W = x.shape
                    xn = norm(x.flatten(2).transpose(1, 2))
                    qkv = F.linear(xn, wi, bi).reshape(B, H * W, 3, nheads, hdim).permute(2, 0, 3, 1, 4)
                    out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, H * W, C)
                    return x + F.linear(out, wo, bo).transpose(1, 2).reshape(B, C, H, W)
                return _fwd

            m.forward = _make_fwd(m.norm, w_i, b_i, w_o, b_o, nh, hd)
            m._sdpa_patched = True
            n_patched += 1
    return n_patched


class FastA0Encoder(nn.Module):
    """Slim A0 VAE Encoder dropping unused logvar channels in quant_conv."""

    def __init__(self, vae: nn.Module, sf: float = 0.18215):
        super().__init__()
        self.encoder = vae.encoder
        self.qc_mean = nn.Conv2d(8, 4, 1)
        with torch.no_grad():
            self.qc_mean.weight.copy_(vae.quant_conv.weight[:4])
            self.qc_mean.bias.copy_(vae.quant_conv.bias[:4])
        self.sf = float(sf)

    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        h = self.qc_mean(self.encoder(rgb01 * 2.0 - 1.0))
        return (h * self.sf).float()


class FastConditioner(nn.Module):
    """Fused de-styled Conditioning Front-End (shared gray + Sobel/Gauss/Lap buffers + CNX FP16 + NS10 Stiefel + Y16)."""

    def __init__(
        self,
        cnx_ext: nn.Module,
        w32_st: torch.Tensor,
        w_luma: torch.Tensor,
        b_luma: torch.Tensor,
        *,
        ns_steps: int = 10,
        use_exact_svd: bool = False,
        cnx_fp16: bool = True,
    ):
        super().__init__()
        self.stage3 = cnx_ext.stage3.half() if cnx_fp16 else cnx_ext.stage3.float()
        self.cnx_fp16 = cnx_fp16
        self.ns_steps = int(ns_steps)
        self.use_exact_svd = bool(use_exact_svd)

        dev = w32_st.device
        self.register_buffer("im_mean", cnx_ext.im_mean.float().to(dev))
        self.register_buffer("im_std", cnx_ext.im_std.float().to(dev))
        self.register_buffer("w32_t", w32_st.float().t().contiguous())
        self.register_buffer("wl", w_luma.float().to(dev).view(1, 4, 1, 1))
        self.register_buffer("bl", b_luma.float().to(dev))
        self.register_buffer(
            "k_gauss",
            torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=dev).view(1, 1, 3, 3) / 16.0,
        )
        self.register_buffer(
            "k_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=dev).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "k_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=dev).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "k_lap",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=dev).view(1, 1, 3, 3),
        )
        self.register_buffer("eye3", torch.eye(w32_st.shape[1], device=dev).unsqueeze(0) * 3.0)

    @staticmethod
    def _lcn(y: torch.Tensor, k: int, eps: float) -> torch.Tensor:
        pad = k // 2
        yp = F.pad(y, (pad, pad, pad, pad), mode="reflect")
        mu = F.avg_pool2d(yp, k, stride=1)
        diff = y - mu
        dp = F.pad(diff.square(), (pad, pad, pad, pad), mode="reflect")
        sd = F.avg_pool2d(dp, k, stride=1).clamp_min(1e-8).sqrt()
        return (diff / (sd + eps)).clamp(-3.0, 3.0) / 3.0

    def forward(self, rgb01: torch.Tensor, z_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = rgb01.shape[0]
        r, g, b = rgb01[:, 0:1].float(), rgb01[:, 1:2].float(), rgb01[:, 2:3].float()
        gray = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)

        # 64x64 4-ch structure map (single pass)
        y128 = F.avg_pool2d(gray, 4, 4)
        y64 = F.avg_pool2d(F.conv2d(F.pad(y128, (1, 1, 1, 1), mode="reflect"), self.k_gauss), 2, 2)
        yp = F.pad(y64, (1, 1, 1, 1), mode="reflect")
        gx, gy = F.conv2d(yp, self.k_x), F.conv2d(yp, self.k_y)
        mag = (gx.square() + gy.square() + 1e-8).sqrt()
        q95 = torch.quantile(mag.flatten(1), 0.95, dim=1).view(B, 1, 1, 1).clamp_min(0.05)
        edge = (mag / q95).clamp(0.0, 1.5) / 1.5 * 2.0 - 1.0
        lap = F.conv2d(yp, self.k_lap)
        lap_rms = lap.flatten(1).square().mean(1).clamp_min(1e-8).sqrt().view(B, 1, 1, 1).clamp_min(0.02)
        c64 = torch.cat(
            [self._lcn(y64, 5, 0.03), self._lcn(y64, 13, 0.04), edge, (lap / (3.0 * lap_rms)).clamp(-1.0, 1.0)],
            dim=1,
        )

        # ConvNeXt Stage-3 grayscale layout (reusing gray)
        x = F.interpolate(gray.repeat(1, 3, 1, 1), size=(256, 256), mode="bilinear", align_corners=False)
        xn = (x - self.im_mean) / self.im_std
        if self.cnx_fp16:
            xn = xn.half()
        p = self.stage3(xn).float()
        pm = p.mean(dim=(2, 3), keepdim=True)
        ps = p.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
        pw = (p - pm) / ps
        pn = pw / pw.norm(dim=1, keepdim=True).clamp_min(1e-6) * (384.0 ** 0.5)
        g16 = pn - pn.mean(dim=(2, 3), keepdim=True)

        # Stiefel Polar Quotient (pure-matmul Newton-Schulz or exact SVD)
        fl = torch.bmm(self.w32_t.unsqueeze(0).expand(B, -1, -1), g16.flatten(2))
        fl = fl - fl.mean(dim=-1, keepdim=True)
        if self.use_exact_svd:
            u, _, vh = torch.linalg.svd(fl, full_matrices=False)
            q16 = (torch.bmm(u, vh) * 16.0).reshape(B, -1, 16, 16)
        else:
            x_ns = fl / fl.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            for _ in range(self.ns_steps):
                x_ns = 0.5 * torch.bmm(self.eye3 - torch.bmm(x_ns, x_ns.transpose(1, 2)), x_ns)
            q16 = (x_ns * 16.0).reshape(B, -1, 16, 16)

        yl = ((z_s.float() * self.wl).sum(dim=1, keepdim=True) + self.bl).clamp(0.0, 1.0)
        y16 = self._lcn(F.avg_pool2d(yl, 4, 4), 3, 0.04)
        return c64, g16, q16, y16


class FastECore(nn.Module):
    """SDPA-patched, nearest-upsampled, FP16-autocast E-Core wrapper with pre-registered priors."""

    def __init__(self, ecore: nn.Module, raw_stats: torch.Tensor, priors_ec: dict[str, torch.Tensor], *, dtype: torch.dtype = torch.float16):
        super().__init__()
        patch_bottleneck_sdpa(ecore)
        self.ecore = ecore
        self.dtype = dtype
        self.register_buffer("raw_stats", raw_stats.float())
        self._pkeys = list(priors_ec.keys())
        for k, v in priors_ec.items():
            self.register_buffer(f"p_{k}", v.float())

    def forward(self, src: torch.Tensor, c64: torch.Tensor, q32: torch.Tensor, tid: torch.Tensor, lam: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        pr = {k: getattr(self, f"p_{k}") for k in self._pkeys}
        if self.dtype == torch.float32:
            return self.ecore(src, c64, q32, tid, lam, self.raw_stats, pr, noise=eps).float()
        with torch.autocast("cuda", dtype=self.dtype):
            return self.ecore(src, c64, q32, tid, lam, self.raw_stats, pr, noise=eps).float()


class HoistedPCoreStep(nn.Module):
    """P-Core Dual-CFG UNet step with trajectory-invariant SPADE, dino_proj, and cond_mlp columns hoisted out."""

    def __init__(self, pcore: nn.Module, *, dtype: torch.dtype = torch.float16):
        super().__init__()
        patch_bottleneck_sdpa(pcore)
        self.m = pcore
        self.dtype = dtype
        self.register_buffer("w_tau", pcore.cond_mlp[0].weight[:, :64].detach().clone())
        self.register_buffer("w_rest", pcore.cond_mlp[0].weight[:, 64:].detach().clone())
        self.register_buffer("b0", pcore.cond_mlp[0].bias.detach().clone())

    def prepare_traj(
        self,
        y16: torch.Tensor,
        q32: torch.Tensor,
        tid: torch.Tensor,
        u_style: torch.Tensor,
        ds4: torch.Tensor,
        dc4: torch.Tensor,
        lsc4: torch.Tensor,
        gamma_val: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = y16.shape[0]
        if gamma_val < 1.0:
            q4 = F.interpolate(F.avg_pool2d(q32, 4), scale_factor=4.0, mode="nearest")
            q4 = q4 - q4.mean(dim=(2, 3), keepdim=True)
            q_eff = gamma_val * q32 + (1.0 - gamma_val) * q4
            y_eff = y16 * gamma_val
        else:
            q_eff, y_eff = q32, y16

        ls = lsc4[:B].view(B, 1, 1, 1)
        s16_on = torch.cat([y_eff * ls, self.m.dino_proj(q_eff) * ls], dim=1)
        s16_off = torch.zeros_like(s16_on)
        s16_pair = torch.cat([s16_on, s16_off], dim=0)

        g1, b1 = self.m.mid1.spade(s16_pair).chunk(2, dim=1)
        g2, b2 = self.m.mid2.spade(s16_pair).chunk(2, dim=1)
        ar = torch.arange(B, device=y16.device)
        idx = torch.cat([ar, ar + B, ar, ar + B], dim=0)
        gc1, bc1, gc2, bc2 = g1[idx], b1[idx], g2[idx], b2[idx]

        sm2 = (~ds4).float().view(4 * B, 1)
        oh = F.one_hot(tid.repeat(4).clamp(0, 4), 5).float() * sm2
        uv = (F.normalize(u_style.repeat(4, 1), dim=-1) * 2.0) * sm2
        lsc_eff = lsc4 * (~dc4).float()
        flags = torch.stack([ds4.float(), 1.0 - lsc_eff], dim=-1)
        g_t = torch.full((4 * B,), float(gamma_val), device=y16.device)
        static = torch.cat([oh, uv, flags, sinusoidal_time_emb(g_t, 16)], dim=-1)
        cond_bias4 = F.linear(static, self.w_rest, self.b0)
        return gc1, bc1, gc2, bc2, cond_bias4

    def _forward_core(
        self,
        z_1b: torch.Tensor,
        t_1b: torch.Tensor,
        cond_bias4: torch.Tensor,
        gc1: torch.Tensor,
        bc1: torch.Tensor,
        gc2: torch.Tensor,
        bc2: torch.Tensor,
    ) -> torch.Tensor:
        m = self.m
        z_4b = z_1b.repeat(4, 1, 1, 1)
        tau_p = F.linear(sinusoidal_time_emb(t_1b, 64), self.w_tau, None).repeat(4, 1)
        cond = m.cond_mlp[3](m.cond_mlp[2](F.silu(tau_p + cond_bias4)))

        h64 = m.in_conv(z_4b)
        for blk in m.enc64:
            h64 = blk(h64, cond)
        h32 = m.down32(h64)
        for blk in m.enc32:
            h32 = blk(h32, cond)
        h16 = m.down16(h32)

        h = m.mid1.norm1(h16)
        h = m.mid1.conv1(m.mid1.act(h * (1.0 + gc1) + bc1))
        gs, bs = m.mid1.film(cond).chunk(2, dim=-1)
        h16 = h16 + m.mid1.conv2(m.mid1.act(m.mid1.norm2(h) * (1.0 + gs[:, :, None, None]) + bs[:, :, None, None]))

        h16 = m.mid_attn(h16)

        h = m.mid2.norm1(h16)
        h = m.mid2.conv1(m.mid2.act(h * (1.0 + gc2) + bc2))
        gs2, bs2 = m.mid2.film(cond).chunk(2, dim=-1)
        h16 = h16 + m.mid2.conv2(m.mid2.act(m.mid2.norm2(h) * (1.0 + gs2[:, :, None, None]) + bs2[:, :, None, None]))

        u32 = m.up32(F.interpolate(h16, scale_factor=2.0, mode="nearest"))
        u32 = m.skip32(torch.cat([u32, h32], dim=1))
        for blk in m.dec32:
            u32 = blk(u32, cond)
        u64 = m.up64(F.interpolate(u32, scale_factor=2.0, mode="nearest"))
        u64 = m.skip64(torch.cat([u64, h64], dim=1))
        for blk in m.dec64:
            u64 = blk(u64, cond)
        return m.out_conv(F.silu(m.out_norm(u64)))

    def forward(
        self,
        z_1b: torch.Tensor,
        t_1b: torch.Tensor,
        cond_bias4: torch.Tensor,
        gc1: torch.Tensor,
        bc1: torch.Tensor,
        gc2: torch.Tensor,
        bc2: torch.Tensor,
    ) -> torch.Tensor:
        if self.dtype == torch.float32:
            return self._forward_core(z_1b, t_1b, cond_bias4, gc1, bc1, gc2, bc2).float()
        with torch.autocast("cuda", dtype=self.dtype):
            return self._forward_core(z_1b, t_1b, cond_bias4, gc1, bc1, gc2, bc2).float()


class FastS10BDecoder(nn.Module):
    """channels_last S10B Decoder wrapper outputting [0, 1] RGB directly."""

    def __init__(self, s10: nn.Module, sf: float = 0.18215, *, dtype: torch.dtype = torch.float16, channels_last: bool = True):
        super().__init__()
        self.s10 = s10.to(dtype=dtype)
        if channels_last:
            self.s10 = self.s10.to(memory_format=torch.channels_last)
        self.dtype = dtype
        self.channels_last = channels_last
        self.sf = float(sf)

    def forward(self, zo: torch.Tensor) -> torch.Tensor:
        z_in = (zo / self.sf).to(dtype=self.dtype)
        if self.channels_last:
            z_in = z_in.contiguous(memory_format=torch.channels_last)
        x = self.s10(z_in)
        return x.float().mul(0.5).add(0.5).clamp(0.0, 1.0)
