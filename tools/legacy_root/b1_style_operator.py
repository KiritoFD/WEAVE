"""路线 B1：内容锚定的多尺度交叉注意力 + 区域重绘算子 T_phi。

按 2026-11 的路线 B 新规格（容量与目标结构优先）：

  * T_phi 目标 5–15M 可训练参数；
  * 多尺度内容特征（1/1, 1/2, 1/4 latent 分辨率）；
  * 每个尺度 2 层、多头参考图交叉注意力：Q 来自内容局部结构，
    K 来自参考图局部结构类型，V 承载颜色/笔触尺度/方向等风格操作；
  * 可学习的区域重绘算子：逐区域预测笔触尺度、方向分布、色块重构强度、
    区域门控；LL 可改但只做有界残差，主结构始终由内容子带承担；
  * 参考图只提供"操作集合"，K/V 是 token set，attention 对其顺序严格置换
    不变；输出空间相位只来自 content。

关键防退化机制：
  * B1 只提供算子、独立风格分布损失（SWD / patch 统计）和筛选器；
  * `b_pseudo_targets.BPseudoTargetProvider` 强制"先通过阶段 1 筛选，才允许
    把 z* 当 FM 端点"，避免 T_phi 在纯 FM 目标下学成恒等映射。

默认不接入 model / flow / trainer；导入本文件不会改变任何现有前向数值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from features import extract_patch_descriptors, patch_self_similarity, sobel_edges
from swd import sliced_wasserstein_distance
from wavelet import dwt2_haar, idwt2_haar


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _grad_xy(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    zp = F.pad(z.float(), (1, 1, 1, 1), mode="replicate")
    gx = 0.5 * (zp[..., 1:-1, 2:] - zp[..., 1:-1, :-2])
    gy = 0.5 * (zp[..., 2:, 1:-1] - zp[..., :-2, 1:-1])
    return gx, gy


def _avg_k(z: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(F.pad(z.float(), (k // 2,) * 4, mode="replicate"), k, stride=1)


def _avg3(z: torch.Tensor) -> torch.Tensor:
    return _avg_k(z, 3)


def _pool_for_selfsim(z: torch.Tensor, max_tokens: int = 256) -> torch.Tensor:
    H, W = z.shape[-2:]
    if H * W <= max_tokens:
        return z
    side = int(math.sqrt(max_tokens))
    return F.adaptive_avg_pool2d(z.float(), (side, side))


def _edge_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ea = sobel_edges(a.float())
    eb = sobel_edges(b.float())
    if ea.shape[-2:] != eb.shape[-2:]:
        eb = F.interpolate(eb, size=ea.shape[-2:], mode="bilinear", align_corners=False)
    xa = ea.flatten(1)
    xb = eb.flatten(1)
    xa = xa - xa.mean(dim=-1, keepdim=True)
    xb = xb - xb.mean(dim=-1, keepdim=True)
    denom = xa.pow(2).sum(-1).sqrt() * xb.pow(2).sum(-1).sqrt()
    return ((xa * xb).sum(-1) / denom.clamp_min(1e-8)).clamp(-1.0, 1.0)


def analytic_structure_map(z: torch.Tensor) -> torch.Tensor:
    """解析式局部结构图（6 通道）：边缘幅值/方向、平滑度、两个尺度的 HF 能量。

    这些量对颜色不敏感，用于 Q（内容结构）与 K（参考结构类型）的显式匹配。
    """
    zf = z.float()
    gx, gy = _grad_xy(zf)
    mag2 = gx.pow(2) + gy.pow(2)
    mag = torch.sqrt(mag2 + 1e-8).mean(dim=1, keepdim=True)
    cos2 = ((gx.pow(2) - gy.pow(2)) / (mag2 + 1e-6)).mean(dim=1, keepdim=True)
    sin2 = ((2.0 * gx * gy) / (mag2 + 1e-6)).mean(dim=1, keepdim=True)
    lp = _avg3(zf)
    resid2 = (zf - lp).pow(2).mean(dim=1, keepdim=True)
    smoothness = 1.0 / (1.0 + resid2)
    hf1 = zf - lp
    hf2 = lp - _avg3(lp)
    e1 = hf1.pow(2).mean(dim=1, keepdim=True)
    e2 = hf2.pow(2).mean(dim=1, keepdim=True)
    return torch.cat(
        [mag, cos2, sin2, smoothness, torch.log1p(100.0 * e1), torch.log1p(100.0 * e2)],
        dim=1,
    ).to(dtype=z.dtype)


def reference_op_map(ref: torch.Tensor) -> torch.Tensor:
    """参考图逐像素风格操作值（2C+4）：局部颜色、相对颜色、HF 能量、方向能量。"""
    ref_f = ref.float()
    color = ref_f
    color_rel = ref_f - ref_f.mean(dim=(2, 3), keepdim=True)
    lp = _avg3(ref_f)
    hf1 = ref_f - lp
    hf2 = lp - _avg3(lp)
    e1 = hf1.pow(2).mean(dim=1, keepdim=True)
    e2 = hf2.pow(2).mean(dim=1, keepdim=True)
    gx, gy = _grad_xy(ref_f)
    gxe = gx.abs().mean(dim=1, keepdim=True)
    gye = gy.abs().mean(dim=1, keepdim=True)
    out = torch.cat(
        [color, color_rel, torch.log1p(100.0 * e1), torch.log1p(100.0 * e2), gxe, gye],
        dim=1,
    )
    return out.to(dtype=ref.dtype)


def reference_global_vector(ref: torch.Tensor) -> torch.Tensor:
    ref_f = ref.float()
    lp = _avg3(ref_f)
    hf1 = ref_f - lp
    hf2 = lp - _avg3(lp)
    gx, gy = _grad_xy(ref_f)
    return torch.cat(
        [
            ref_f.mean(dim=(2, 3)),
            ref_f.std(dim=(2, 3)),
            hf1.pow(2).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
            hf2.pow(2).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
            gx.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
            gy.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
        ],
        dim=1,
    ).to(dtype=ref.dtype)


def _resize_map(m: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    if m.shape[-2:] == hw:
        return m
    return F.adaptive_avg_pool2d(m.float(), hw).to(dtype=m.dtype)


def _tokens(m: torch.Tensor, side: int) -> torch.Tensor:
    p = F.adaptive_avg_pool2d(m.float(), (side, side))
    return p.flatten(2).transpose(1, 2).contiguous().to(dtype=m.dtype)


def _oriented_kernel(k: int, theta: float) -> torch.Tensor:
    c = (k - 1) / 2.0
    y, x = torch.meshgrid(
        torch.arange(k, dtype=torch.float32) - c,
        torch.arange(k, dtype=torch.float32) - c,
        indexing="ij",
    )
    sigma = max(k / 3.0, 1.0)
    g = torch.exp(-(x.pow(2) + y.pow(2)) / (2.0 * sigma * sigma))
    ker = -(x * math.cos(theta) + y * math.sin(theta)) / (sigma * sigma) * g
    ker = ker - ker.mean()
    return ker / ker.abs().sum().clamp_min(1e-6)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class B1OperatorConfig:
    latent_channels: int = 4
    # 多尺度编码器 / 解码器
    widths: Tuple[int, int, int] = (96, 192, 288)
    encoder_blocks: Tuple[int, int, int] = (2, 2, 2)
    attention_heads: Tuple[int, int, int] = (4, 8, 8)
    attention_layers: Tuple[int, int, int] = (2, 2, 2)
    attention_dropout: float = 0.0
    token_side: int = 16
    decoder_hidden: int = 96
    # 区域重绘
    num_directions: int = 8
    num_brush_scales: int = 3
    direction_kernel: int = 5
    block_kernel: int = 5
    # T_phi 有界残差 / 门控
    style_gate_init: float = 0.2
    ll_shift_max: float = 2.0
    hf_shift_max: float = 0.8
    use_hh: bool = True
    style_dim: int = 32
    # 与 latent patch 统计对齐（供 loss / screener 使用）
    patch_size: int = 4
    patch_stride: int = 4
    extra: Dict = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.widths) != 3 or len(self.encoder_blocks) != 3:
            raise ValueError("widths / encoder_blocks 必须都是 3 个尺度")
        if len(self.attention_heads) != 3 or len(self.attention_layers) != 3:
            raise ValueError("attention_heads / attention_layers 必须都是 3 个尺度")
        for w in self.widths:
            if w % 8 != 0:
                raise ValueError(f"width={w} 必须能被 GroupNorm 的 8 组整除")


# ---------------------------------------------------------------------------
# 多尺度内容编码器
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class MultiScaleContentEncoder(nn.Module):
    """1/1, 1/2, 1/4 三个尺度的内容特征。内容与参考图共享该编码器。"""

    def __init__(self, cfg: B1OperatorConfig) -> None:
        super().__init__()
        C = cfg.latent_channels
        w0, w1, w2 = cfg.widths
        n0, n1, n2 = cfg.encoder_blocks
        self.stem = nn.Conv2d(C, w0, 3, padding=1)
        self.blocks0 = nn.Sequential(*[ResBlock(w0) for _ in range(n0)])
        self.down01 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.blocks1 = nn.Sequential(*[ResBlock(w1) for _ in range(n1)])
        self.down12 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.blocks2 = nn.Sequential(*[ResBlock(w2) for _ in range(n2)])

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        f0 = self.blocks0(self.stem(z.float()))
        f1 = self.blocks1(self.down01(f0))
        f2 = self.blocks2(self.down12(f1))
        return [f0, f1, f2]


# ---------------------------------------------------------------------------
# 多头 / 多层参考交叉注意力
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        need_weights: bool = False,
    ):
        qn = self.norm_q(q)
        kn = self.norm_k(k)
        if need_weights:
            a, w = self.attn(qn, kn, v, need_weights=True, average_attn_weights=True)
        else:
            a, w = self.attn(qn, kn, v, need_weights=False)
        q = q + a
        q = q + self.ff(self.norm_ff(q))
        return q, w


class CrossAttentionStack(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim, heads, dropout) for _ in range(layers)]
        )

    def forward(self, q, k, v, *, return_attention: bool = False):
        w = None
        for i, block in enumerate(self.blocks):
            q, wi = block(
                q, k, v, need_weights=return_attention and i == len(self.blocks) - 1
            )
            if wi is not None:
                w = wi
        return q, w


# ---------------------------------------------------------------------------
# 区域重绘算子（T_phi）
# ---------------------------------------------------------------------------

class RegionRepaintRenderer(nn.Module):
    """在 Haar 子带上执行区域级重绘。

    可解释的区域参数（由 RegionRepaintOperator 预测）：
      * block_strength: 色块重构强度（LL 局部平涂）
      * direction:      8 个方向的笔触权重
      * brush_scale:    3 个尺度的笔触权重
      * region_gate:    区域是否重绘

    结构约束：LL/HF 都以 content 子带为底，参考图只通过区域参数与全局操作
    （`global_op`）进入；HF 残差由 content HF 的有向滤波响应生成，不读参考 HF。
    """

    def __init__(self, cfg: B1OperatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        C = cfg.latent_channels
        hf_ch = 3 * C
        hidden = 64
        side = int(cfg.direction_kernel)

        self.dir_conv = nn.Conv2d(
            hf_ch, hf_ch * cfg.num_directions,
            kernel_size=side, padding=side // 2, groups=hf_ch, bias=False,
        )
        self._init_directional_bank()

        self.ll_enc = nn.Conv2d(2 * C, hidden, 3, padding=1)
        self.ll_mix = nn.Sequential(nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.ll_out = nn.Conv2d(hidden, C, 1)

        self.hf_enc = nn.Conv2d(3 * hf_ch, hidden, 3, padding=1)
        self.hf_mix = nn.Sequential(nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.hf_out = nn.Conv2d(hidden, hf_ch, 1)

        self.global_mlp = nn.Linear(int(cfg.extra.get("style_dim", 32)), hidden)
        for m in (self.ll_out, self.hf_out):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        init = float(min(max(cfg.style_gate_init, 1e-4), 0.999))
        self.style_gate = nn.Parameter(torch.tensor(math.atanh(init)))

    def _init_directional_bank(self) -> None:
        K = int(self.cfg.num_directions)
        w = self.dir_conv.weight  # (hf_ch*K, 1, k, k)
        with torch.no_grad():
            for d in range(K):
                theta = math.pi * d / K
                ker = _oriented_kernel(int(self.cfg.direction_kernel), theta)
                for c in range(w.shape[0] // K):
                    w[c * K + d, 0] = ker

    def gate_value(self) -> torch.Tensor:
        return torch.tanh(self.style_gate).detach()

    def forward(
        self,
        content_latent: torch.Tensor,
        params: Dict[str, torch.Tensor],
        global_op: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        C = int(cfg.latent_channels)
        ll, lh, hl, hh = dwt2_haar(content_latent)
        hw = ll.shape[-2:]
        B = ll.shape[0]
        g = self.global_mlp(global_op.float().to(dtype=ll.dtype)).view(B, -1, 1, 1)
        gate = torch.tanh(self.style_gate).to(dtype=ll.dtype)
        s = gate * params["region_gate"]  # (B,1,h,w)

        # LL：内容 LL 做局部色块平涂，再叠加有界风格残差。
        ll_blocked = F.avg_pool2d(
            F.pad(ll, (cfg.block_kernel // 2,) * 4, mode="replicate"),
            cfg.block_kernel, stride=1,
        )
        strength = params["block_strength"]
        ll_flat = ll + strength * (ll_blocked - ll)
        ll_feat = self.ll_mix(self.ll_enc(torch.cat([ll_flat, ll], dim=1)) + g)
        ll_res = torch.tanh(self.ll_out(ll_feat))
        ll_out = ll_flat + s * ll_res * float(cfg.ll_shift_max)

        # HF：content HF -> 有向滤波 bank -> 区域方向/尺度加权 -> 有界残差。
        hf = torch.cat([lh, hl, hh], dim=1)  # (B,3C,h,w)
        resp = self.dir_conv(hf)  # (B,3C*K,h,w)
        K = int(cfg.num_directions)
        resp = resp.view(B, 3 * C, K, *hw)
        direction = params["direction"].unsqueeze(1)  # (B,1,K,h,w)
        hf_dir = (resp * direction).sum(dim=2)

        scales = [hf_dir, _avg_k(hf_dir, 3), _avg_k(hf_dir, 5)][: int(cfg.num_brush_scales)]
        brush = params["brush_scale"]  # (B,Ks,h,w)
        hf_multi = torch.zeros_like(hf_dir)
        for k in range(len(scales)):
            hf_multi = hf_multi + brush[:, k : k + 1] * scales[k]

        hf_feat = self.hf_mix(self.hf_enc(torch.cat([hf, hf_dir, hf_multi], dim=1)) + g)
        hf_res = torch.tanh(self.hf_out(hf_feat))
        hf_out = hf + s * hf_res * float(cfg.hf_shift_max)

        lh_out = hf_out[:, 0:1]
        hl_out = hf_out[:, 1:2]
        hh_out = hf_out[:, 2:3] if cfg.use_hh else hh
        z_star = idwt2_haar(ll_out, lh_out, hl_out, hh_out)
        return {
            "z_star": z_star.to(dtype=content_latent.dtype),
            "ll_in": ll,
            "ll_flat": ll_flat,
            "ll_out": ll_out,
            "lh_out": lh_out,
            "hl_out": hl_out,
            "hh_out": hh_out,
            "ll_delta": ll_out - ll,
            "hf_delta": hf_out - hf,
        }


class RegionRepaintOperator(nn.Module):
    """解码多尺度特征 -> 逐区域参数 -> 小波域重绘。"""

    def __init__(self, cfg: B1OperatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        w0, w1, w2 = cfg.widths
        hidden = int(cfg.decoder_hidden)
        self.up2 = nn.Conv2d(w2, w1, 3, padding=1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(w1 * 2, w1, 3, padding=1), nn.GroupNorm(8, w1), nn.SiLU()
        )
        self.up1 = nn.Conv2d(w1, w0, 3, padding=1)
        self.fuse0 = nn.Sequential(
            nn.Conv2d(w0 * 2, w0, 3, padding=1), nn.GroupNorm(8, w0), nn.SiLU()
        )
        self.res0 = ResBlock(w0)
        self.param_conv = nn.Conv2d(w0, hidden, 3, padding=1)
        self.param_act = nn.SiLU()

        self.block_logits = nn.Conv2d(hidden, 1, 1)
        self.region_logits = nn.Conv2d(hidden, 1, 1)
        self.dir_logits = nn.Conv2d(hidden, int(cfg.num_directions), 1)
        self.brush_logits = nn.Conv2d(hidden, int(cfg.num_brush_scales), 1)
        # 初始：轻微色块化、区域默认开启，方向/尺度均匀。
        nn.init.zeros_(self.block_logits.weight)
        nn.init.constant_(self.block_logits.bias, -2.0)
        nn.init.zeros_(self.region_logits.weight)
        nn.init.constant_(self.region_logits.bias, 2.0)
        nn.init.zeros_(self.dir_logits.weight)
        nn.init.zeros_(self.dir_logits.bias)
        nn.init.zeros_(self.brush_logits.weight)
        nn.init.zeros_(self.brush_logits.bias)

        self.renderer = RegionRepaintRenderer(cfg)
        self.style_dim = int(cfg.extra.get("style_dim", 32))

    def forward(
        self,
        content_latent: torch.Tensor,
        feats: Sequence[torch.Tensor],
        global_op: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        f0, f1, f2 = feats
        d1 = self.fuse1(torch.cat([F.interpolate(self.up2(f2), size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], dim=1))
        d0 = self.fuse0(torch.cat([F.interpolate(self.up1(d1), size=f0.shape[-2:], mode="bilinear", align_corners=False), f0], dim=1))
        d0 = self.res0(d0)
        ll_size = (content_latent.shape[-2] // 2, content_latent.shape[-1] // 2)
        h = self.param_act(self.param_conv(F.adaptive_avg_pool2d(d0, ll_size)))

        params = {
            "block_strength": torch.sigmoid(self.block_logits(h)),
            "region_gate": torch.sigmoid(self.region_logits(h)),
            "direction": torch.softmax(self.dir_logits(h).float(), dim=1).to(dtype=h.dtype),
            "brush_scale": torch.softmax(self.brush_logits(h).float(), dim=1).to(dtype=h.dtype),
        }
        out = self.renderer(content_latent, params, global_op)
        out["region_params"] = params
        return out


# ---------------------------------------------------------------------------
# B1 主算子
# ---------------------------------------------------------------------------

class B1StyleOperator(nn.Module):
    def __init__(self, cfg: B1OperatorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or B1OperatorConfig()
        self.cfg.validate()
        C = int(self.cfg.latent_channels)
        self.style_dim = int(self.cfg.style_dim)
        self.cfg.extra["style_dim"] = self.style_dim

        self.encoder = MultiScaleContentEncoder(self.cfg)
        widths = list(self.cfg.widths)

        # Q/K 显式拼接解析结构图；V 拼接参考风格操作图。
        self.q_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(w + 6, w), nn.LayerNorm(w)) for w in widths
        ])
        self.k_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(w + 6, w), nn.LayerNorm(w)) for w in widths
        ])
        self.v_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(w + (2 * C + 4), w), nn.LayerNorm(w)) for w in widths
        ])
        self.attn_stacks = nn.ModuleList([
            CrossAttentionStack(w, h, l, self.cfg.attention_dropout)
            for w, h, l in zip(widths, self.cfg.attention_heads, self.cfg.attention_layers)
        ])
        self.level_fuse = nn.ModuleList([
            nn.Conv2d(w, w, 1) for w in widths
        ])
        self.fusion_gate = nn.ParameterList([
            nn.Parameter(torch.tensor(0.10)) for _ in widths
        ])

        self.global_proj = nn.Sequential(
            nn.Linear(2 * C + 4, self.style_dim),
            nn.SiLU(),
            nn.Linear(self.style_dim, self.style_dim),
        )
        self.repaint = RegionRepaintOperator(self.cfg)

    # -- 参考分支 ---------------------------------------------------------
    def _encode_reference(
        self,
        reference_latent: torch.Tensor,
        *,
        reference_token_permutation: torch.Tensor | None = None,
    ) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        feats = self.encoder(reference_latent)
        desc = analytic_structure_map(reference_latent)
        ops = reference_op_map(reference_latent)
        ks, vs = [], []
        for i, w in enumerate(self.cfg.widths):
            hw = feats[i].shape[-2:]
            k_tokens = _tokens(
                torch.cat([_resize_map(feats[i], hw), _resize_map(desc, hw)], dim=1),
                self.cfg.token_side,
            )
            v_tokens = _tokens(
                torch.cat([_resize_map(feats[i], hw), _resize_map(ops, hw)], dim=1),
                self.cfg.token_side,
            )
            k = self.k_proj[i](k_tokens.float()).to(dtype=reference_latent.dtype)
            v = self.v_proj[i](v_tokens.float()).to(dtype=reference_latent.dtype)
            if reference_token_permutation is not None:
                perm = reference_token_permutation.to(device=k.device, dtype=torch.long)
                k = k[:, perm].contiguous()
                v = v[:, perm].contiguous()
            ks.append(k)
            vs.append(v)
        global_op = self.global_proj(reference_global_vector(reference_latent).float())
        return {"k": ks, "v": vs, "global": global_op}

    # -- 内容分支 ---------------------------------------------------------
    def _encode_content(self, content_latent: torch.Tensor) -> Dict:
        feats = self.encoder(content_latent)
        desc = analytic_structure_map(content_latent)
        qs = []
        for i in range(3):
            hw = feats[i].shape[-2:]
            q_tokens = _tokens(
                torch.cat([_resize_map(feats[i], hw), _resize_map(desc, hw)], dim=1),
                self.cfg.token_side,
            )
            qs.append(self.q_proj[i](q_tokens.float()).to(dtype=content_latent.dtype))
        return {"q": qs, "feats": feats}

    # -- 前向 -------------------------------------------------------------
    def forward(
        self,
        content_latent: torch.Tensor,
        reference_latent: torch.Tensor | None = None,
        *,
        global_style_op: torch.Tensor | None = None,
        reference_token_permutation: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> Dict:
        B = content_latent.shape[0]
        content = self._encode_content(content_latent)
        feats = content["feats"]
        fused = list(feats)
        attention = None
        ref_tokens = 0

        if reference_latent is not None:
            ref = self._encode_reference(
                reference_latent,
                reference_token_permutation=reference_token_permutation,
            )
            global_op = ref["global"]
            if global_op.shape[0] == 1 and B > 1:
                global_op = global_op.expand(B, -1)
            ref_tokens = int(ref["k"][0].shape[1])
            for i in range(3):
                k, v = ref["k"][i], ref["v"][i]
                if k.shape[0] == 1 and B > 1:
                    k = k.expand(B, -1, -1)
                    v = v.expand(B, -1, -1)
                q = content["q"][i]
                out_tokens, w = self.attn_stacks[i](
                    q, k, v, return_attention=return_attention and i == 0
                )
                if w is not None:
                    attention = w
                side = int(self.cfg.token_side)
                w_i = int(self.cfg.widths[i])
                attn_map = out_tokens.transpose(1, 2).reshape(B, w_i, side, side)
                attn_map = F.interpolate(
                    attn_map, size=feats[i].shape[-2:], mode="bilinear", align_corners=False
                )
                fused[i] = feats[i] + torch.tanh(self.fusion_gate[i]) * self.level_fuse[i](attn_map)
        else:
            # 只有离线全局风格操作时，逐区域操作置零；仍由 renderer 生成风格残差。
            ref_tokens = 0
            global_op = (
                global_style_op
                if global_style_op is not None
                else content_latent.new_zeros(B, self.style_dim)
            )

        if global_style_op is not None:
            global_op = global_style_op.to(device=content_latent.device, dtype=content_latent.dtype)
            if global_op.shape[0] == 1 and B > 1:
                global_op = global_op.expand(B, -1)

        out = self.repaint(content_latent, fused, global_op)
        result = {
            "z_star": out["z_star"],
            "region_params": out["region_params"],
            "global_op": global_op,
            "ll_in": out["ll_in"],
            "ll_out": out["ll_out"],
            "lh_out": out["lh_out"],
            "hl_out": out["hl_out"],
            "hh_out": out["hh_out"],
            "ll_delta": out["ll_delta"],
            "hf_delta": out["hf_delta"],
            "gate": self.repaint.renderer.gate_value(),
            "content_tokens": int(self.cfg.token_side ** 2),
            "reference_tokens": ref_tokens,
        }
        if return_attention:
            result["attention"] = attention
        return result

    def attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """诊断：参考注意力的平均熵，越小说明区域匹配越尖锐。"""
        p = attention.clamp_min(1e-8)
        return -(p * p.log()).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# 独立风格分布监督
# ---------------------------------------------------------------------------

class B1StyleDistributionLoss(nn.Module):
    """阶段 1 的独立风格监督：风格分布匹配 + 内容结构保持。"""

    def __init__(
        self,
        *,
        patch_size: int = 4,
        patch_stride: int = 4,
        num_projections: int = 32,
        w_style_swd: float = 1.0,
        w_style_moment: float = 0.5,
        w_content_selfsim: float = 1.0,
        w_content_edge: float = 0.5,
        w_ll_hf: float = 0.1,
        w_phase_copy: float = 0.0,
        phase_copy_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.num_projections = int(num_projections)
        self.w_style_swd = float(w_style_swd)
        self.w_style_moment = float(w_style_moment)
        self.w_content_selfsim = float(w_content_selfsim)
        self.w_content_edge = float(w_content_edge)
        self.w_ll_hf = float(w_ll_hf)
        self.w_phase_copy = float(w_phase_copy)
        self.phase_copy_threshold = float(phase_copy_threshold)

    def _desc(self, z: torch.Tensor) -> torch.Tensor:
        d = extract_patch_descriptors(z.float(), self.patch_size, self.patch_stride)
        return d.reshape(-1, d.shape[-1])

    def _style_moments(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        af, bf = a.float(), b.float()
        loss = F.l1_loss(af.mean(dim=(2, 3)), bf.mean(dim=(2, 3)))
        loss = loss + F.l1_loss(af.std(dim=(2, 3)), bf.std(dim=(2, 3)))
        ha = af - _avg3(af)
        hb = bf - _avg3(bf)
        loss = loss + F.l1_loss(ha.pow(2).mean(dim=(2, 3)), hb.pow(2).mean(dim=(2, 3)))
        gxa, gya = _grad_xy(af)
        gxb, gyb = _grad_xy(bf)
        prof_a = torch.cat([gxa.abs().mean(dim=(2, 3)), gya.abs().mean(dim=(2, 3))], dim=-1)
        prof_b = torch.cat([gxb.abs().mean(dim=(2, 3)), gyb.abs().mean(dim=(2, 3))], dim=-1)
        return loss + F.l1_loss(prof_a, prof_b)

    def _content_selfsim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        sa = patch_self_similarity(_pool_for_selfsim(a))
        sb = patch_self_similarity(_pool_for_selfsim(b))
        return F.l1_loss(sa, sb)

    def forward(
        self,
        z_star: torch.Tensor,
        z_content: torch.Tensor,
        z_style: torch.Tensor,
        *,
        ll_delta: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        swd = sliced_wasserstein_distance(
            self._desc(z_star), self._desc(z_style),
            num_projections=self.num_projections, p=2,
        )
        moment = self._style_moments(z_star, z_style)
        selfsim = self._content_selfsim(z_star, z_content)
        edge = F.l1_loss(sobel_edges(z_star.float()), sobel_edges(z_content.float()))
        total = (
            self.w_style_swd * swd
            + self.w_style_moment * moment
            + self.w_content_selfsim * selfsim
            + self.w_content_edge * edge
        )
        logs: Dict[str, torch.Tensor] = {
            "loss_style_swd": swd.detach(),
            "loss_style_moment": moment.detach(),
            "loss_content_selfsim": selfsim.detach(),
            "loss_content_edge": edge.detach(),
        }
        if ll_delta is not None and self.w_ll_hf > 0.0:
            lhf = (ll_delta.float() - _avg3(ll_delta.float())).pow(2).mean()
            total = total + self.w_ll_hf * lhf
            logs["loss_ll_hf"] = lhf.detach()
        if self.w_phase_copy > 0.0:
            corr = _edge_correlation(z_star, z_style).mean()
            pen = F.relu(corr - self.phase_copy_threshold).pow(2)
            total = total + self.w_phase_copy * pen
            logs["loss_phase_copy"] = pen.detach()
            logs["edge_corr_with_ref"] = corr.detach()
        logs["loss"] = total
        return logs


# ---------------------------------------------------------------------------
# 伪目标筛选
# ---------------------------------------------------------------------------

class StyleTargetScreener:
    def __init__(
        self,
        *,
        min_style_gain: float = 0.05,
        min_content_keep: float = 0.60,
        max_reference_copy: float = 0.50,
        patch_size: int = 4,
        patch_stride: int = 4,
        num_projections: int = 32,
    ) -> None:
        self.min_style_gain = float(min_style_gain)
        self.min_content_keep = float(min_content_keep)
        self.max_reference_copy = float(max_reference_copy)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.num_projections = int(num_projections)

    def _desc(self, z: torch.Tensor) -> torch.Tensor:
        d = extract_patch_descriptors(z.float(), self.patch_size, self.patch_stride)
        return d.reshape(-1, d.shape[-1])

    @torch.no_grad()
    def score(self, z_star: torch.Tensor, z_content: torch.Tensor, z_ref: torch.Tensor) -> Dict:
        # 两次 SWD 使用同一组随机投影，避免 Monte Carlo 噪声让恒等映射"看起来"有增益。
        def _proj_gen() -> torch.Generator:
            gen = torch.Generator(device=z_star.device)
            gen.manual_seed(20260101)
            return gen

        d_c = sliced_wasserstein_distance(
            self._desc(z_content), self._desc(z_ref),
            num_projections=self.num_projections, generator=_proj_gen(),
        )
        d_o = sliced_wasserstein_distance(
            self._desc(z_star), self._desc(z_ref),
            num_projections=self.num_projections, generator=_proj_gen(),
        )
        gain = float((d_c - d_o) / (d_c + 1e-8))
        sim_o = patch_self_similarity(_pool_for_selfsim(z_star.float()))
        sim_c = patch_self_similarity(_pool_for_selfsim(z_content.float()))
        content_keep = float(1.0 - (sim_o - sim_c).abs().mean() / 2.0)
        edge_corr = float(_edge_correlation(z_star, z_ref).mean())
        return {
            "style_distance_content": float(d_c),
            "style_distance_output": float(d_o),
            "style_gain": gain,
            "content_keep": content_keep,
            "reference_copy_corr": edge_corr,
            "passed": bool(
                gain >= self.min_style_gain
                and content_keep >= self.min_content_keep
                and edge_corr <= self.max_reference_copy
            ),
        }


# ---------------------------------------------------------------------------
# 阶段 3 结构探针
# ---------------------------------------------------------------------------

@torch.no_grad()
def reference_swap_probe(
    operator: B1StyleOperator,
    content_latent: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
) -> Dict[str, float]:
    out_a = operator(content_latent, reference_a)["z_star"]
    out_b = operator(content_latent, reference_b)["z_star"]
    delta = (out_a - out_b).abs()
    scale = out_a.abs().mean().clamp_min(1e-6)
    return {
        "output_delta_abs": float(delta.mean()),
        "output_delta_rel": float(delta.mean() / scale),
        "output_delta_max": float(delta.max()),
    }


@torch.no_grad()
def reference_permutation_probe(
    operator: B1StyleOperator,
    content_latent: torch.Tensor,
    reference_latent: torch.Tensor,
    *,
    permutation: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Dict[str, float]:
    n = int(operator.cfg.token_side ** 2)
    if permutation is None:
        permutation = torch.randperm(n, generator=generator)
    base = operator(content_latent, reference_latent)["z_star"]
    shuffled = operator(
        content_latent, reference_latent, reference_token_permutation=permutation
    )["z_star"]
    delta = (base - shuffled).abs()
    return {
        "token_count": n,
        "output_delta_abs": float(delta.mean()),
        "output_delta_max": float(delta.max()),
    }


def build_b1_operator(**kwargs) -> B1StyleOperator:
    cfg_keys = set(B1OperatorConfig.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_keys}
    return B1StyleOperator(B1OperatorConfig(**cfg_kwargs))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "B1OperatorConfig",
    "MultiScaleContentEncoder",
    "CrossAttentionStack",
    "RegionRepaintOperator",
    "B1StyleOperator",
    "B1StyleDistributionLoss",
    "StyleTargetScreener",
    "reference_swap_probe",
    "reference_permutation_probe",
    "build_b1_operator",
    "count_parameters",
]
