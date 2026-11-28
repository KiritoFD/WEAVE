"""V36 Open-Ref 1-NFE Reference E-Core (~11.8M params, strictly Zero-DINO & Zero Class Labels).

Bridges source painting latents into target-style structural-brushwork drafts before frozen V36 45M P-Core:
  1. Exact mathematical identity at lambda=0 (z_out = z_src).
  2. Reference-A3 (I2 Chroma+Luma) progressive interpolation z0 = z_src + lambda * (z_A3 - z_src).
  3. Permutation-invariant K=8 spatial-agnostic Style Prototype Bank extracted from z_ref via slot attention
     (no positional coords -> cannot copy reference spatial layout), paired with 1561-D ConvNeXt+Oklab+Haar stats.
  4. Gated source high-freq structure channels (P_Y(z_src) and 4-ch c_64) multiplied by (1 - 0.85 * lambda).
  5. 16x16 bottleneck Cross-Attention between source layout/features and the K=8 style prototypes + Spatial AdaGN dec32/64.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v28_unified.student_models import SpatialRoutedResBlock
from exp.v29_biglittle.pcore_model import (
    BottleneckAttention,
    PureStyleResBlock,
    SPADEStyleBlock,
    sinusoidal_time_emb,
)
from exp.v36_art_foundation.eval_v36 import reconstruct_reference_a3
from exp.v36_art_foundation.models import StyleMLPEncoder
from src.utils.structure import latent_to_structure_64


class StylePrototypeBank(nn.Module):
    """Extract K spatial-agnostic style prototypes from reference latent z_ref (zero spatial copying)."""

    def __init__(self, in_ch: int = 4, proto_dim: int = 128, out_dim: int = 288, num_protos: int = 8):
        super().__init__()
        self.num_protos = num_protos
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, proto_dim, 3, stride=2, padding=1),
            nn.GroupNorm(16, proto_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(proto_dim, proto_dim, 3, padding=1),
        )
        self.slot_logits = nn.Conv2d(proto_dim, num_protos, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(proto_dim),
            nn.Linear(proto_dim, out_dim),
        )

    def forward(self, z_ref: torch.Tensor) -> torch.Tensor:
        h = self.feat(z_ref.float())  # (B, D, 16, 16)
        B, D, H, W = h.shape
        attn = self.slot_logits(h).flatten(2).softmax(dim=-1)  # (B, K, HW), permutation-invariant pool
        h_flat = h.flatten(2).transpose(1, 2)  # (B, HW, D)
        protos = torch.bmm(attn, h_flat)  # (B, K, D)
        return self.proj(protos)  # (B, K, out_dim)


class BottleneckStyleCrossAttn(nn.Module):
    """Source-queried cross-attention over K spatial-agnostic reference style prototypes."""

    def __init__(self, dim: int = 288, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.norm_q = nn.GroupNorm(16, dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        hd = self.heads
        d_h = C // hd
        q = self.q_proj(self.norm_q(h)).view(B, hd, d_h, H * W).transpose(2, 3)  # (B, hd, HW, d_h)
        kv = self.kv_proj(self.norm_kv(protos)).view(B, protos.shape[1], 2, hd, d_h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, hd, K, d_h)
        attn = (torch.matmul(q, k.transpose(-1, -2)) * self.scale).softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(2, 3).contiguous().view(B, C, H, W)
        return h + self.gate.tanh() * self.out_proj(out)


class ReferenceECoreNet(nn.Module):
    """~11.8M 1-NFE Open-Ref Pre-Editor matching V36 Zero-DINO latent + Stiefel Q32 conventions."""

    def __init__(
        self,
        stat_mean: torch.Tensor | None = None,
        stat_std: torch.Tensor | None = None,
        *,
        w0: int = 128,
        w1: int = 192,
        w2: int = 288,
        cond_dim: int = 192,
        num_protos: int = 8,
    ):
        super().__init__()
        self.style_mlp = StyleMLPEncoder(1561, 512, 256, stat_mean, stat_std)
        self.proto_bank = StylePrototypeBank(4, 128, w2, num_protos=num_protos)
        self.cond_mlp = nn.Sequential(
            nn.Linear(32 + 256, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

        # 4 (z0) + 4 (gated P_Y(z_src)) + 4 (gated c_64) + 4 (noise) = 16
        self.in_conv = nn.Conv2d(16, w0, 3, padding=1)
        self.enc64 = nn.ModuleList([PureStyleResBlock(w0, cond_dim), PureStyleResBlock(w0, cond_dim)])
        self.down32 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc32 = nn.ModuleList([PureStyleResBlock(w1, cond_dim), PureStyleResBlock(w1, cond_dim)])
        self.down16 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)

        q_ch = 64
        self.q_proj = nn.Sequential(
            nn.Conv2d(32, 96, 1),
            nn.GroupNorm(16, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, q_ch, 1),
        )
        self.mid1 = SPADEStyleBlock(w2, q_ch, cond_dim)
        self.mid_self_attn = BottleneckAttention(w2, heads=8)
        self.mid_proto_attn = BottleneckStyleCrossAttn(w2, heads=8)
        self.mid2 = SPADEStyleBlock(w2, q_ch, cond_dim)

        route_ch = 128
        self.route = nn.Sequential(BottleneckAttention(32, heads=8), nn.Conv2d(32, route_ch, 1))
        self.route_gate = nn.Linear(cond_dim, route_ch)

        self.up32 = nn.Conv2d(w2, w1, 3, padding=1)
        self.skip32 = nn.Conv2d(w1 * 2, w1, 1)
        self.dec32 = nn.ModuleList([
            SpatialRoutedResBlock(w1, cond_dim, route_ch),
            SpatialRoutedResBlock(w1, cond_dim, route_ch),
        ])

        self.up64 = nn.Conv2d(w1, w0, 3, padding=1)
        self.skip64 = nn.Conv2d(w0 * 2, w0, 1)
        self.dec64 = nn.ModuleList([
            SpatialRoutedResBlock(w0, cond_dim, route_ch),
            SpatialRoutedResBlock(w0, cond_dim, route_ch),
        ])

        self.out_norm = nn.GroupNorm(16, w0)
        self.out_conv = nn.Conv2d(w0, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        z_src: torch.Tensor,
        z_ref: torch.Tensor,
        q32: torch.Tensor,
        stat_ref: torch.Tensor,
        strength: torch.Tensor,
        priors: dict[str, torch.Tensor],
        *,
        noise: torch.Tensor | None = None,
        alpha_y: float = 1.0,
        alpha_c: float = 1.0,
    ) -> torch.Tensor:
        """Single 1-NFE pre-edit forward pass with strict identity at strength == 0."""
        B = z_src.shape[0]
        lam = strength.to(z_src.device).float().view(B).clamp(0.0, 1.25)
        amount = lam.view(B, 1, 1, 1)

        z_a3 = reconstruct_reference_a3(z_src.float(), z_ref.float(), priors, mode="I2",
                                        alpha_y=alpha_y, alpha_c=alpha_c)
        z0 = z_src.float() + amount.clamp(0.0, 1.0) * (z_a3 - z_src.float())

        p_y = priors["P_Y"].to(z_src.device).float()
        luma = torch.einsum("ij,bjhw->bihw", p_y, z_src.float())
        c64 = latent_to_structure_64(z_src.float(), priors["luma_w"], priors["luma_b"])
        gate_src = (1.0 - 0.85 * amount).clamp(0.0, 1.0)
        eps = torch.zeros_like(z0) if noise is None else noise.float()
        noise_ch = 0.12 * amount * eps

        e_s = self.style_mlp(stat_ref.float())
        cond = self.cond_mlp(torch.cat([sinusoidal_time_emb(lam, 32), e_s], dim=-1))
        protos = self.proto_bank(z_ref.float())

        q_f = q32.float()
        s16_q = self.q_proj(q_f)
        r16 = self.route(q_f) * self.route_gate(cond).tanh()[:, :, None, None]
        r32 = F.interpolate(r16, size=(32, 32), mode="bilinear", align_corners=False)
        r64 = F.interpolate(r16, size=(64, 64), mode="bilinear", align_corners=False)

        inp = torch.cat([z0, luma * gate_src, c64 * gate_src, noise_ch], dim=1)
        h64 = self.in_conv(inp)
        for blk in self.enc64:
            h64 = blk(h64, cond)
        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)
        h16 = self.down16(h32)

        h16 = self.mid1(h16, s16_q, cond)
        h16 = self.mid_self_attn(h16)
        h16 = self.mid_proto_attn(h16, protos)
        h16 = self.mid2(h16, s16_q, cond)

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        for blk in self.dec32:
            u32 = blk(u32, cond, r32)

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond, r64)

        delta = self.out_conv(F.silu(self.out_norm(u64)))
        return z0 + amount * delta
