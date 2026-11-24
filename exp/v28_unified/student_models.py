"""Phase 3 Student Models: UnifiedEditor (6.7M, M0/M1) and UnifiedHybridUNet (~11.5M, M2/M3) + Pixel Bridge."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18.ccf_model import BottleneckAttention, SPADEStyleBlock, sinusoidal_time_emb
from exp.v19.fiber_duality import UnnormalizedVGGFeatureExtractor, apply_latent_chroma_shift, extract_color_fiber_27d
from exp.v20.slf_flow import PureStyleResBlock
from exp.v27_single.unified.model import UnifiedEditor
from exp.v27_single.unified.train import conditioning_stats, haar_signature
from src.utils import decode_s10, load_s10_decoder
from src.utils.runtime import bf16_autocast
from src.utils.structure import dwt2_haar


class SpatialRoutedResBlock(nn.Module):
    """PureStyleResBlock + zero-initialized Spatial AdaGN modulation from Q32 route map."""

    def __init__(self, width: int, cond_dim: int, route_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(16, width, affine=False)
        self.norm2 = nn.GroupNorm(16, width, affine=False)
        self.film1 = nn.Linear(cond_dim, width * 2)
        self.film2 = nn.Linear(cond_dim, width * 2)
        self.smod1 = nn.Conv2d(route_ch, width * 2, 1)
        self.smod2 = nn.Conv2d(route_ch, width * 2, 1)
        nn.init.zeros_(self.smod1.weight)
        nn.init.zeros_(self.smod1.bias)
        nn.init.zeros_(self.smod2.weight)
        nn.init.zeros_(self.smod2.bias)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, r_map: torch.Tensor) -> torch.Tensor:
        g1, b1 = self.film1(cond)[:, :, None, None].chunk(2, 1)
        dg1, db1 = self.smod1(r_map).chunk(2, 1)
        h = self.conv1(F.silu(self.norm1(x) * (1.0 + g1 + dg1.tanh()) + b1 + db1))
        g2, b2 = self.film2(cond)[:, :, None, None].chunk(2, 1)
        dg2, db2 = self.smod2(r_map).chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h) * (1.0 + g2 + dg2.tanh()) + b2 + db2))
        return x + h


class UnifiedHybridUNet(nn.Module):
    """M2/M3 (~11.5M): 64->32->16->32->64 Residual Hybrid U-Net with:
    - Gated source high-freq direct inputs (P_Y(z_s) and c_64 multiplied by (1 - 0.80 * lambda))
    - Lightweight noise channel 0.12 * lambda * eps
    - 16x16 pure Q32 bottleneck with BottleneckAttention + SPADE
    - Spatial AdaGN routing at dec32 & dec64
    - Strict outer z0(lambda) + lambda * delta mathematical identity at lambda=0.
    """

    def __init__(self, stats_dim: int, w0: int = 128, w1: int = 192, w2: int = 288, cond_dim: int = 192):
        super().__init__()
        self.stats = nn.Linear(stats_dim, 64)
        self.cond = nn.Sequential(nn.Linear(5 + 32 + 64, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        # 4 (z0) + 4 (gated luma P_Y(z_s)) + 4 (gated c64) + 4 (noise) = 16
        self.in_conv = nn.Conv2d(16, w0, 3, padding=1)
        self.enc64 = nn.ModuleList([PureStyleResBlock(w0, cond_dim), PureStyleResBlock(w0, cond_dim)])
        self.down32 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc32 = nn.ModuleList([PureStyleResBlock(w1, cond_dim), PureStyleResBlock(w1, cond_dim)])
        self.down16 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)

        # Pure 16x16 Q32 bottleneck
        q_ch = 64
        self.q_proj = nn.Sequential(
            nn.Conv2d(32, 96, 1), nn.GroupNorm(16, 96), nn.SiLU(inplace=True), nn.Conv2d(96, q_ch, 1),
        )
        self.mid1 = SPADEStyleBlock(w2, q_ch, cond_dim)
        self.mid_attn = BottleneckAttention(w2, heads=8)
        self.mid2 = SPADEStyleBlock(w2, q_ch, cond_dim)

        # Spatial Q32 routing map -> dec32 & dec64 Spatial AdaGN
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
        self.out = nn.Conv2d(w0, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        source: torch.Tensor,
        c64: torch.Tensor,
        q32: torch.Tensor,
        tid: torch.Tensor,
        strength: torch.Tensor,
        raw_stats: torch.Tensor,
        priors: dict,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        amount = strength[:, None, None, None]
        aligned = apply_latent_chroma_shift(source, tid, priors, alpha=1.0)
        z0 = source + 0.35 * amount * (aligned - source)
        luma = torch.einsum("ij,bjhw->bihw", priors["P_Y"], source)

        # Gated source high-freq direct connection: releases 80% edge lock at lambda=1 while keeping lambda=0 exact
        gate_src = (1.0 - 0.80 * amount).clamp(0.0, 1.0)
        eps = torch.zeros_like(source) if noise is None else noise
        noise_ch = 0.12 * amount * eps

        style = self.stats(raw_stats[tid])
        cond = self.cond(torch.cat((F.one_hot(tid, 5).float(), sinusoidal_time_emb(strength, 32), style), 1))

        s16_q = self.q_proj(q32)
        r16 = self.route(q32) * self.route_gate(cond).tanh()[:, :, None, None]
        r32 = F.interpolate(r16, size=(32, 32), mode="bilinear", align_corners=False)
        r64 = F.interpolate(r16, size=(64, 64), mode="bilinear", align_corners=False)

        inp = torch.cat((z0, luma * gate_src, c64 * gate_src, noise_ch), dim=1).contiguous(
            memory_format=torch.channels_last
        )
        h64 = self.in_conv(inp)
        for blk in self.enc64:
            h64 = blk(h64, cond)
        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)
        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16_q, cond)
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16_q, cond)

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat((u32, h32), dim=1))
        for blk in self.dec32:
            u32 = blk(u32, cond, r32)

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat((u64, h64), dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond, r64)

        delta = self.out(F.silu(self.out_norm(u64)))
        return z0 + amount * delta


class PixelBridgeV28:
    """S10-BF16 orthogonal VGG-SWD + VGG/Oklab moment perceptual bridge."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.decoder = load_s10_decoder(device).to(torch.bfloat16).requires_grad_(False)
        self.vgg = UnnormalizedVGGFeatureExtractor().to(device).eval().requires_grad_(False)
        self.layers: list[torch.Tensor] = []
        for layer in (self.vgg.s1, self.vgg.s2, self.vgg.s3):
            layer.register_forward_hook(self._capture)
        self.scale: torch.Tensor | None = None

    def _capture(self, _m, _inp, out):
        self.layers.append(out)

    def encode(self, z: torch.Tensor, grad: bool) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, allow_tf32=True), bf16_autocast(self.device):
            rgb = decode_s10(self.decoder, z, dtype=torch.bfloat16, batch=len(z),
                             enable_grad=grad, output_cpu=False)
            self.layers.clear()
            features = self.vgg(rgb, multi_scale=False).float()
            tokens = tuple(F.adaptive_avg_pool2d(x, 16).float().flatten(2).transpose(1, 2) for x in self.layers)
            self.layers.clear()
        return features, tokens

    def loss(
        self,
        pred: torch.Tensor,
        goal: torch.Tensor,
        proj: tuple[torch.Tensor, ...],
        *,
        oklab_weight: float = 0.25,
        swd_weight: float = 0.15,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current, tokens = self.encode(pred, grad=True)
        with torch.no_grad():
            target, ref_tokens = self.encode(goal, grad=False)
        moments_ps = F.smooth_l1_loss(
            (current - target) / self.scale, torch.zeros_like(current), beta=0.20, reduction="none"
        ).mean(dim=1)
        swd_ps = sum(
            ((a @ p).sort(dim=1).values - (b @ p).sort(dim=1).values).abs().mean(dim=(1, 2))
            for a, b, p in zip(tokens, ref_tokens, proj)
        ) / len(proj)
        if sample_weights is not None:
            swd_ps = swd_ps * sample_weights.to(swd_ps.device, dtype=swd_ps.dtype)
        # Note: current[:27] is Oklab 27D * 3.0 inside UnnormalizedVGGFeatureExtractor
        return (oklab_weight * moments_ps + swd_weight * swd_ps).mean()


def build_student_model(arch: str, stats_dim: int, device: str = "cuda") -> nn.Module:
    if arch == "flat":
        return UnifiedEditor(stats_dim).to(device).to(memory_format=torch.channels_last)
    if arch == "unet":
        return UnifiedHybridUNet(stats_dim).to(device).to(memory_format=torch.channels_last)
    raise ValueError(f"Unknown student arch: {arch}")
