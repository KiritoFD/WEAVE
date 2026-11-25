"""CLEAVE V33 Models & Math Primitives:
1. Q64+ Orthogonal Complement Stiefel Quotient (with rank-degeneracy handling) & Partial Whitening Residual.
2. Multi-scale ConvNeXt-Tiny Stage-3 Grayscale Extractor (256px -> 16x16, 512px -> 32x32) + Stiefel projection.
3. Q64ResidualAdapter (~2.3M, 16x16 bottleneck only, zero-init, zero CFG leak).
4. SemanticControlAdapter (~6.9M, 16x16 & 32x32 injection only, zero 64x64 skips, zero-init, zero CFG leak).
5. Unified V33 trajectory integrator supporting Y16 gating, Q64+, Q32_hi, and 6/8/12 DPM-2M/RK2 steps.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18.ccf_model import BottleneckAttention, SPADEStyleBlock, sinusoidal_time_emb
from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor
from src.utils.runtime import bf16_autocast
from src.utils.structure import rgb_to_gray


def _orthonormal_complement_rows(
    existing: torch.Tensor,
    n_need: int,
    seed_offset: int = 0,
) -> torch.Tensor:
    """Construct n_need unit-norm rows in R^N orthogonal to existing (K, N) and constant vector 1."""
    N = existing.shape[-1]
    dev = existing.device
    ones = torch.ones(1, N, device=dev, dtype=torch.float64) / math.sqrt(float(N))
    basis = torch.cat([ones, existing.to(torch.float64)], dim=0)  # (1+K, N)
    g = torch.Generator(device="cpu").manual_seed(20261133 + seed_offset)
    cand = torch.randn(n_need + 16, N, generator=g).to(device=dev, dtype=torch.float64)
    # Project out existing basis twice for numerical precision
    for _ in range(2):
        cand = cand - (cand @ basis.T) @ basis
    # QR on transposed candidates -> orthonormal columns -> transpose to rows
    q_cols, r_mat = torch.linalg.qr(cand.T, mode="reduced")
    diag = r_mat.diagonal().abs()
    valid = q_cols[:, diag > 1e-8].T
    if valid.shape[0] < n_need:
        raise RuntimeError(f"Insufficient null-space dimension for complement: got {valid.shape[0]} < {n_need}")
    return (valid[:n_need] * math.sqrt(float(N))).to(torch.float32)


def stiefel_q64_plus(
    p16: torch.Tensor,
    w32: torch.Tensor,
    wb32: torch.Tensor,
    *,
    q32: torch.Tensor | None = None,
    rank_tol: float = 1e-5,
) -> torch.Tensor:
    """Construct Q64+ = [Q32; Q_b] preserving exact Q32 and (1/N) Q64+ Q64+^T == I_64.

    Handles rank-deficient inputs by completing degenerate complement rows strictly
    inside null([1, Q32, valid_Q_b]) so Q32 is untouched and Gram orthogonality holds.
    """
    B, _, H, W = p16.shape
    N = H * W
    sqrt_n = math.sqrt(float(N))
    q32_map = stiefel_polar_quotient(p16, w32) if q32 is None else q32.to(device=p16.device, dtype=torch.float32)
    q32_flat = q32_map.reshape(B, w32.shape[1], N)

    wb = wb32.to(device=p16.device, dtype=torch.float32)
    zb = torch.einsum("dk,bdp->bkp", wb, p16.float().flatten(2))
    zb = zb - zb.mean(dim=-1, keepdim=True)
    # Project out Q32 row space: Z_b^perp = Z_b (I_N - Q^T Q / N)
    zb_perp = zb - torch.bmm(zb, q32_flat.transpose(1, 2) / float(N)) @ q32_flat

    U, S, Vh = torch.linalg.svd(zb_perp, full_matrices=False)
    qb_flat = sqrt_n * torch.bmm(U, Vh)

    min_s = S.min(dim=-1).values
    if bool((min_s <= rank_tol).any()):
        k_b = wb.shape[1]
        for b in range(B):
            if float(min_s[b].item()) <= rank_tol:
                valid_mask = S[b] > rank_tol
                n_val = int(valid_mask.sum().item())
                if n_val > 0:
                    v_rows = Vh[b, valid_mask] * sqrt_n
                    # Re-orthogonalize valid rows against [1, Q32] for strict stability
                    ones = torch.ones(1, N, device=p16.device, dtype=torch.float32) / sqrt_n
                    q_unit = q32_flat[b] / sqrt_n
                    v_unit = v_rows / sqrt_n
                    v_unit = v_unit - (v_unit @ ones.T) @ ones - (v_unit @ q_unit.T) @ q_unit
                    qv, _ = torch.linalg.qr(v_unit.T, mode="reduced")
                    v_clean = qv.T * sqrt_n
                    exist = torch.cat([q32_flat[b] / sqrt_n, v_clean / sqrt_n], dim=0)
                    rem = _orthonormal_complement_rows(exist, k_b - n_val, seed_offset=b)
                    qb_b = torch.cat([v_clean, rem], dim=0)
                else:
                    qb_b = _orthonormal_complement_rows(q32_flat[b] / sqrt_n, k_b, seed_offset=b)
                qb_flat[b] = qb_b

    qb_map = qb_flat.reshape(B, wb.shape[1], H, W)
    return torch.cat([q32_map, qb_map], dim=1)


def partial_whitening_residual(
    p16: torch.Tensor,
    w32: torch.Tensor,
    *,
    q32: torch.Tensor | None = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute RMS-calibrated partial-whitening residual (exponent alpha=0.5) relative to full Q32."""
    B, _, H, W = p16.shape
    N = H * W
    q32_map = stiefel_polar_quotient(p16, w32) if q32 is None else q32.to(device=p16.device, dtype=torch.float32)
    q32_flat = q32_map.reshape(B, w32.shape[1], N)

    w = w32.to(device=p16.device, dtype=torch.float32)
    fl = torch.einsum("dk,bdp->bkp", w, p16.float().flatten(2))
    fl = fl - fl.mean(dim=-1, keepdim=True)
    U, S, Vh = torch.linalg.svd(fl, full_matrices=False)
    s_pow = S.clamp_min(0.0).pow(float(alpha))
    pw = torch.bmm(U * s_pow.unsqueeze(1), Vh)
    pw_rms = pw.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    pw = pw / pw_rms

    res = pw - q32_flat
    res = res - res.mean(dim=-1, keepdim=True)
    res_rms = res.square().mean(dim=(1, 2), keepdim=True).sqrt()
    res = torch.where(res_rms > 1e-6, res / res_rms.clamp_min(1e-6), torch.zeros_like(res))
    return res.reshape(B, w32.shape[1], H, W)


class ConvNeXtMultiScaleExtractor(ConvNeXtTinyStage3Extractor):
    """Reuses frozen ConvNeXt-Tiny Stage-3 weights for 256px->16x16 and 512px->32x32 grayscale layouts."""

    @torch.no_grad()
    def extract_gray_layout_at(self, rgb01: torch.Tensor, size: int) -> torch.Tensor:
        gray3 = rgb_to_gray(rgb01.float().clamp(0.0, 1.0)).repeat(1, 3, 1, 1)
        x = F.interpolate(gray3, size=(size, size), mode="bilinear", align_corners=False)
        xn = (x - self.im_mean.to(x.device)) / self.im_std.to(x.device)
        p = self.stage3(xn).float()
        mean = p.mean(dim=(2, 3), keepdim=True)
        std = p.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
        p_whiten = (p - mean) / std
        p_norm = p_whiten / p_whiten.norm(dim=1, keepdim=True).clamp_min(1e-6) * math.sqrt(384.0)
        return p_norm - p_norm.mean(dim=(2, 3), keepdim=True)

    @torch.no_grad()
    def extract_gray_layout_32(self, rgb01: torch.Tensor) -> torch.Tensor:
        return self.extract_gray_layout_at(rgb01, size=512)


def extract_cnx_q32_hi(p32_gray: torch.Tensor, w_stiefel_hi: torch.Tensor) -> torch.Tensor:
    """Project (B, 384, 32, 32) grayscale ConvNeXt Stage-3 features onto Stiefel frame -> (B, 32, 32, 32)."""
    B, _, H, W = p32_gray.shape
    N = H * W
    w = w_stiefel_hi.to(device=p32_gray.device, dtype=torch.float32)
    fl = torch.einsum("dk,bdp->bkp", w, p32_gray.float().flatten(2))
    fl = fl - fl.mean(dim=-1, keepdim=True)
    U, _, Vh = torch.linalg.svd(fl, full_matrices=False)
    q = torch.bmm(U, Vh) * math.sqrt(float(N))
    return q.reshape(B, w.shape[1], H, W)


class Q64ResidualAdapter(nn.Module):
    """~2.3M zero-init 16x16 bottleneck residual adapter consuming Q64+ / Q32+partial-whiten residual."""

    def __init__(self, in_q_ch: int = 64, w_mid16: int = 432, cond_dim: int = 320, inner_ch: int = 256):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Conv2d(in_q_ch, 160, 1),
            nn.GroupNorm(16, 160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, 128, 3, padding=1),
        )
        self.in_proj = nn.Conv2d(w_mid16, inner_ch, 1)
        self.block = SPADEStyleBlock(inner_ch, 129, cond_dim)
        self.out_norm = nn.GroupNorm(16, inner_ch)
        self.out_conv = nn.Conv2d(inner_ch, w_mid16, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        h16: torch.Tensor,
        q64: torch.Tensor,
        y16: torch.Tensor,
        cond: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_masked = q64 * active_mask
        s_map = torch.cat([y16 * active_mask, self.q_proj(q_masked)], dim=1)
        h = self.block(self.in_proj(h16), s_map, cond)
        delta = self.out_conv(F.silu(self.out_norm(h)))
        return delta * (active_mask > 0.0).float()


class PCoreWithQ64Adapter(nn.Module):
    """Wraps frozen V32 PCoreSLFUNet with zero-init Q64ResidualAdapter at the 16x16 bottleneck."""

    def __init__(self, pcore: PCoreSLFUNet, adapter: Q64ResidualAdapter):
        super().__init__()
        self.pcore = pcore
        self.adapter = adapter
        self.use_gamma = pcore.use_gamma

    def forward(
        self,
        z_tau: torch.Tensor,
        y16_lcn: torch.Tensor,
        q32_16: torch.Tensor,
        style_id: torch.Tensor,
        u_style: torch.Tensor,
        tau: torch.Tensor,
        *,
        q_extra: torch.Tensor | None = None,
        drop_style: torch.Tensor | None = None,
        drop_content: torch.Tensor | None = None,
        layout_scale: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)
        if q_extra is None:
            q_extra = torch.zeros_like(q32_16)

        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        q_in, y_in = q32_16, y16_lcn
        qe_in = q_extra
        if self.pcore.use_gamma:
            g_val = torch.ones(B, device=dev, dtype=torch.float32) if gamma is None else gamma.to(dev).float().view(B).clamp(0.0, 1.0)
            g4 = g_val.view(B, 1, 1, 1)
            q4 = F.avg_pool2d(q_in, 4).repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
            q4 = q4 - q4.mean(dim=(2, 3), keepdim=True)
            qe4 = F.avg_pool2d(qe_in, 4).repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
            qe4 = qe4 - qe4.mean(dim=(2, 3), keepdim=True)
            if self.training:
                q4 = q4 + (1.0 - g4) * 0.10 * torch.randn_like(q4)
            q_in = g4 * q_in + (1.0 - g4) * q4
            qe_in = g4 * qe_in + (1.0 - g4) * qe4
            y_in = y_in * g4

        d_feat = self.pcore.dino_proj(q_in * cmask4) * lscale4
        s16 = torch.cat([y_in * lscale4, d_feat], dim=1)

        oh = F.one_hot(style_id.clamp(0, self.pcore.N_STYLE - 1), self.pcore.N_STYLE).float() * smask2
        uv = (F.normalize(u_style, dim=-1) * 2.0 if self.pcore.normalize_u else u_style) * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond_parts = [temb, oh, uv, flags]
        if self.pcore.use_gamma:
            cond_parts.append(sinusoidal_time_emb(g_val, 16))
        cond = self.pcore.cond_mlp(torch.cat(cond_parts, dim=-1))

        h64 = self.pcore.in_conv(z_tau)
        for blk in self.pcore.enc64:
            h64 = blk(h64, cond)

        h32 = self.pcore.down32(h64)
        for blk in self.pcore.enc32:
            h32 = blk(h32, cond)

        h16 = self.pcore.down16(h32)
        h16 = self.pcore.mid1(h16, s16, cond)
        h16 = h16 + self.adapter(h16, torch.cat([q_in, qe_in], dim=1), y_in, cond, cmask4 * lscale4)
        h16 = self.pcore.mid_attn(h16)
        h16 = self.pcore.mid2(h16, s16, cond)

        u32 = self.pcore.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.pcore.skip32(torch.cat([u32, h32], dim=1))
        for blk in self.pcore.dec32:
            u32 = blk(u32, cond)

        u64 = self.pcore.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.pcore.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.pcore.dec64:
            u64 = blk(u64, cond)

        return self.pcore.out_conv(F.silu(self.pcore.out_norm(u64)))


class SemanticControlAdapter(nn.Module):
    """~6.9M residual control branch injecting ONLY at P-Core 16x16 & 32x32 layers (zero 64x64 skip!)."""

    def __init__(
        self,
        d_q16: int = 32,
        d_q32: int = 32,
        w32: int = 144,
        w16: int = 256,
        pcore_w32: int = 288,
        pcore_w16: int = 432,
        cond_dim: int = 320,
    ):
        super().__init__()
        self.q16_proj = nn.Sequential(
            nn.Conv2d(d_q16, 128, 1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        self.q32_proj = nn.Sequential(
            nn.Conv2d(d_q32, 128, 1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        self.z_to_32 = nn.Conv2d(4, w32, 3, stride=2, padding=1)
        self.enc32 = nn.ModuleList([
            SPADEStyleBlock(w32, 128, cond_dim),
            SPADEStyleBlock(w32, 128, cond_dim),
        ])
        self.down16 = nn.Conv2d(w32, w16, 3, stride=2, padding=1)
        self.mid16 = nn.ModuleList([
            SPADEStyleBlock(w16, 128, cond_dim),
            SPADEStyleBlock(w16, 128, cond_dim),
        ])
        self.mid_attn = BottleneckAttention(w16, heads=8)
        self.up32 = nn.Conv2d(w16, w32, 3, padding=1)
        self.skip32 = nn.Conv2d(w32 * 2, w32, 1)
        self.dec32 = SPADEStyleBlock(w32, 128, cond_dim)

        self.zero_16 = nn.Conv2d(w16, pcore_w16, 1)
        self.zero_32 = nn.Conv2d(w32, pcore_w32, 1)
        nn.init.zeros_(self.zero_16.weight)
        nn.init.zeros_(self.zero_16.bias)
        nn.init.zeros_(self.zero_32.weight)
        nn.init.zeros_(self.zero_32.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        z_t: torch.Tensor,
        cond: torch.Tensor,
        q16: torch.Tensor,
        q32_hi: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s32 = self.q32_proj(q32_hi * active_mask)
        s16 = self.q16_proj(q16 * active_mask)

        h32 = self.z_to_32(z_t)
        for blk in self.enc32:
            h32 = blk(h32, s32, cond)

        h16 = self.down16(h32)
        h16 = self.mid16[0](h16, s16, cond)
        h16 = self.mid_attn(h16)
        h16 = self.mid16[1](h16, s16, cond)

        d32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        d32 = self.skip32(torch.cat([d32, h32], dim=1))
        d32 = self.dec32(d32, s32, cond)

        gate = (active_mask > 0.0).float()
        return self.zero_16(h16) * gate, self.zero_32(d32) * gate


class PCoreWithSemanticControl(nn.Module):
    """V33 P-Core with SemanticControlAdapter injecting ONLY at 16x16 & 32x32 layers."""

    def __init__(self, pcore: PCoreSLFUNet, ctrl: SemanticControlAdapter):
        super().__init__()
        self.pcore = pcore
        self.ctrl = ctrl
        self.use_gamma = pcore.use_gamma

    def forward(
        self,
        z_tau: torch.Tensor,
        y16_lcn: torch.Tensor,
        q32_16: torch.Tensor,
        style_id: torch.Tensor,
        u_style: torch.Tensor,
        tau: torch.Tensor,
        *,
        q32_hi: torch.Tensor | None = None,
        drop_style: torch.Tensor | None = None,
        drop_content: torch.Tensor | None = None,
        layout_scale: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)
        if q32_hi is None:
            q32_hi = F.interpolate(q32_16, size=(32, 32), mode="bilinear", align_corners=False)

        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        q_in, y_in = q32_16, y16_lcn
        if self.pcore.use_gamma:
            g_val = torch.ones(B, device=dev, dtype=torch.float32) if gamma is None else gamma.to(dev).float().view(B).clamp(0.0, 1.0)
            g4 = g_val.view(B, 1, 1, 1)
            q4 = F.avg_pool2d(q_in, 4).repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
            q4 = q4 - q4.mean(dim=(2, 3), keepdim=True)
            if self.training:
                q4 = q4 + (1.0 - g4) * 0.10 * torch.randn_like(q4)
            q_in = g4 * q_in + (1.0 - g4) * q4
            y_in = y_in * g4

        d_feat = self.pcore.dino_proj(q_in * cmask4) * lscale4
        s16 = torch.cat([y_in * lscale4, d_feat], dim=1)

        oh = F.one_hot(style_id.clamp(0, self.pcore.N_STYLE - 1), self.pcore.N_STYLE).float() * smask2
        uv = (F.normalize(u_style, dim=-1) * 2.0 if self.pcore.normalize_u else u_style) * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond_parts = [temb, oh, uv, flags]
        if self.pcore.use_gamma:
            cond_parts.append(sinusoidal_time_emb(g_val, 16))
        cond = self.pcore.cond_mlp(torch.cat(cond_parts, dim=-1))

        active_mask = cmask4 * lscale4
        delta16, delta32 = self.ctrl(z_tau, cond, q32_16, q32_hi, active_mask)

        h64 = self.pcore.in_conv(z_tau)
        for blk in self.pcore.enc64:
            h64 = blk(h64, cond)

        h32 = self.pcore.down32(h64)
        for blk in self.pcore.enc32:
            h32 = blk(h32, cond)

        h16 = self.pcore.down16(h32)
        h16 = self.pcore.mid1(h16, s16, cond)
        h16 = h16 + delta16
        h16 = self.pcore.mid_attn(h16)
        h16 = self.pcore.mid2(h16, s16, cond)

        u32 = self.pcore.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.pcore.skip32(torch.cat([u32, h32], dim=1))
        u32 = u32 + delta32
        for blk in self.pcore.dec32:
            u32 = blk(u32, cond)

        u64 = self.pcore.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.pcore.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.pcore.dec64:
            u64 = blk(u64, cond)

        return self.pcore.out_conv(F.silu(self.pcore.out_norm(u64)))


@torch.no_grad()
def integrate_v33_trajectory(
    model: nn.Module,
    ze: torch.Tensor | None,
    eps: torch.Tensor,
    y16: torch.Tensor,
    q32: torch.Tensor,
    tid: torch.Tensor,
    u_style: torch.Tensor,
    *,
    tau0: float | torch.Tensor,
    ws: float,
    wc: float,
    wsp: float,
    rescale_phi: float = 0.0,
    gamma: float = 1.0,
    lscale: float = 0.92,
    steps: int = 6,
    solver: str = "dpm2m",
    q_extra: torch.Tensor | None = None,
    q32_hi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dual-CFG ODE integrator supporting per-sample tau0, Q64+, and 16/32 Semantic Control."""
    dev = eps.device
    B = len(eps)
    if isinstance(tau0, torch.Tensor):
        tau_v = tau0.to(device=dev, dtype=torch.float32).view(B).clamp(0.0, 1.0)
    else:
        tau_v = torch.full((B,), max(0.0, float(tau0)), device=dev, dtype=torch.float32)

    if ze is None or bool((tau_v <= 0.0).all()):
        tau_v = torch.zeros(B, device=dev, dtype=torch.float32)
        z = eps.clone()
    else:
        t4 = tau_v.view(B, 1, 1, 1)
        z = (1.0 - t4) * eps + t4 * ze

    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls = torch.full((B,), float(lscale), device=dev)
    ls0 = torch.zeros(B, device=dev)
    g_t = torch.full((B,), float(gamma), device=dev) if getattr(model, "use_gamma", True) else None

    y4 = y16.repeat(4, 1, 1, 1)
    q4 = q32.repeat(4, 1, 1, 1)
    s4 = tid.repeat(4)
    u4 = u_style.repeat(4, 1)
    ds4 = torch.cat([f0, f0, f1, f1], 0)
    dc4 = torch.cat([f0, f1, f0, f1], 0)
    lsc4 = torch.cat([ls, ls0, ls, ls0], 0)
    g4 = g_t.repeat(4) if g_t is not None else None
    qe4 = q_extra.repeat(4, 1, 1, 1) if q_extra is not None else None
    qh4 = q32_hi.repeat(4, 1, 1, 1) if q32_hi is not None else None

    def _vel(zz: torch.Tensor, t_vec: torch.Tensor, prog: float) -> torch.Tensor:
        extra_kw = {}
        if qe4 is not None:
            extra_kw["q_extra"] = qe4
        if qh4 is not None:
            extra_kw["q32_hi"] = qh4
        with bf16_autocast(str(dev)):
            v_sc, v_s0, v_0c, v_00 = model(
                zz.repeat(4, 1, 1, 1), y4, q4, s4, u4, t_vec.repeat(4),
                drop_style=ds4, drop_content=dc4, layout_scale=lsc4, gamma=g4, **extra_kw,
            ).float().chunk(4, 0)
        gl = min(1.0, max(0.0, (prog - 0.35) / 0.35))
        ws_j = ws - 0.6 * gl
        wsp_eff = wsp * (0.4 + 0.6 * gl)
        vr = v_00 + wc * (v_0c - v_00) + ws_j * (v_sc - v_0c) + wsp_eff * (v_s0 - v_00)
        if rescale_phi > 0.0:
            s_ref = v_sc.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
            s_raw = vr.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
            vr = rescale_phi * vr * (s_ref / s_raw) + (1.0 - rescale_phi) * vr
        return vr

    grid = torch.linspace(0.0, 1.0, steps + 1, device=dev)
    prev_v = None
    for i in range(steps):
        p0, p1 = float(grid[i].item()), float(grid[i + 1].item())
        dp = p1 - p0
        dt = dp * (1.0 - tau_v).view(B, 1, 1, 1)
        t0 = tau_v + p0 * (1.0 - tau_v)
        v1 = _vel(z, t0, p0)
        if solver == "dpm2m" and prev_v is not None:
            dp_prev = float(grid[i].item() - grid[i - 1].item())
            r = dp / max(dp_prev, 1e-5)
            z = z + dt * (v1 + 0.5 * r * (v1 - prev_v))
        else:
            pm = 0.5 * (p0 + p1)
            tm = tau_v + pm * (1.0 - tau_v)
            v2 = _vel(z + 0.5 * dt * v1, tm, pm)
            z = z + dt * v2
        prev_v = v1
    return z
