"""V25 机制 A: 空间自适应风格混合门控 (A0/A1/A2/A3) —— B01 数学与代码正确性实现.

数学形态 (docs/19_pull_together/02_mechanism_plan.md 机制 A, agent2.md §二):

    z_out = z_P1 + γ · M ⊙ Δz,      Δz = z_D6 − z_P1
    M   : 16×16 单通道空间门控 (上采样到 64×64 后逐元素作用于 latent)

门控归一化次序铁律 (clamp 会破坏 mean=1, 次序不可换):
    1) M_raw = 1 + A·tanh(logits)          # A=GATE_AMPLITUDE, 有界 [1−A, 1+A]
    2) M = clamp(M_raw, 0, GATE_CAP)       # 逐点安全界 [0, 2.5]; 低端可触底 → 空间均值偏离 1
    3) M ← M / max(mean(M), max(M)/CAP)    # clamp 之后必须再次均值归一
       分母取 max(·) 保证归一后逐点仍 ≤ CAP (否则裸 mean-norm 会重新突破上限);
       可行域内 (mean ≥ max/CAP) 精确恢复 mean=1; cap 触底域退化为 mean<1 且 max=CAP
       (诚实反映"模型想降总能量但被逐点上限截住"); 全零角回退恒等门 M≡1.
    16×16 原生网格与 64×64 latent 网格各归一一次 (双线性上采样不精确保均值).

相同总体编辑能量对照 (agent2.md §二要求比较相近 E 而非相同名义 γ):
    edit_energy = ||γ·M⊙Δz||₂ / ||Δz||₂  (逐任务; A0 恒等于 γ)
    E(γ) 对固定 M 线性 → per-task 能量匹配 γ*_i = γ_target / m_rms_i,
    m_rms_i = ||M⊙Δz_i||₂ / ||Δz_i||₂ (energy_matched_gamma).
    A2/A3 结果同时记录名义 γ 与实测 E 两个量.

四臂: A0_global   全局线性混合基线 (M≡1, 复用 Round 1 曲线 C 语义, 教师口径改为本 spec);
      A1_heuristic 固定启发式 M = 1+κ(1−ē), ē = c_64 梯度密度 per-sample 归一 (低细节区给更大 γ);
      A2_learned  GateNet (~0.3M ≤0.5M, 末层零初始化 → 初始 M≡1 即 A0 恒等热启动);
      A3_smooth   A2 + 0.05·TV(M) + 高内容边缘两侧门控连续性惩罚.

教师定义固定铁律: TeacherSpec 与 exp/v25.precompute_train_latents.TEACHER_SPEC 逐字段一致,
训练与评测共用 (assert_same_teacher), 并写入结果 JSON; 教师目标一律消费
runs/v25/cache/train_latents_4400.pt 的 {z_p1_tr, z_d6_tr, teacher_spec}.

用法 (4090, 本任务只交付不启动):
  python -m exp.v25.mechanism_a --arm A2 --gamma 0.55 --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import EVAL_SEED, OUT, ROOT, STYLES, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc
from src.utils import decode_s10, load_s10_decoder
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v23.p1_midfreq_loss import extract_convnext_mid_spatial
from exp.v24.p0_m0_m2 import V20_CKPT, compute_prism1_latents
from exp.v25.precompute_train_latents import TEACHER_SPEC as _PRECOMPUTED_TEACHER_SPEC
from exp.v25.r1_inference_ablation import eval_latent_v25, run_core_flow
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline

# ---- 机制 A 常量 (预注册, 不得在臂间私下改动) ----
GATE_AMPLITUDE = 1.5     # tanh 振幅: M_raw ∈ [1−1.5, 1+1.5] = [−0.5, 2.5] → clamp 低端可触底
GATE_CAP = 2.5           # 门控逐点上限
TV_WEIGHT = 0.05         # A3: TV(M) 系数 (02_mechanism_plan.md 预注册)
EDGE_WEIGHT = 0.05       # A3: 高内容边缘连续性惩罚系数 (计划未定值, 取与 TV 同量级, 见 RISKS)
EDGE_QUANTILE = 0.75     # 高内容边缘掩码分位数
A_SEED = 20263510        # 全臂同 seed → 共享采样顺序 (r2_train_ablation 工程规范)
CAL_SEED = 20263511      # λ_c 标定探针子集 seed
A_MICRO = 16             # 02 计划 A.3: micro=16 (S10 可微解码重载)
A_EFFECTIVE = 64         # 有效 batch = 16×4 accum
TRAIN_GAMMA_DEFAULT = 0.55  # Round 1 曲线 C 工作点 (DINO-C ≈ 0.7321 ≈ 0.73 水位)
# 注: 评测侧 solver 一律取 TeacherSpec.solver (唯一权威), 不在本地重复定义 D6H_SOLVER


# ============================================================================
# 1) 门控数学: 归一化次序 + 能量核算 (纯函数, CPU 可测)
# ============================================================================

def normalize_gate(M_raw: torch.Tensor, cap: float = GATE_CAP, eps: float = 1e-6) -> torch.Tensor:
    """clamp → mean-norm 的唯一合法次序 (见模块 docstring 次序铁律).

    输入 M_raw: (B,1,H,W) 任意实值 (tanh 偏移后的门控原始场).
    输出: 逐点 ∈ [0, cap]; 可行域内 spatial mean == 1; cap 触底域 mean<1 且 max==cap;
          全零 (M_raw ≤ 0 everywhere) → 恒等门 M≡1 (能量守恒唯一自洽解, 不可归一零场).
    """
    M = M_raw.clamp(0.0, cap)  # 次序 2: clamp 在前 —— 此步破坏 mean=1
    mx = M.amax(dim=(2, 3), keepdim=True)
    mean = M.mean(dim=(2, 3), keepdim=True)
    # 次序 3: mean-norm 在后; 分母下界 max/cap 防止归一后重新突破上限
    denom = torch.maximum(mean, mx / cap).clamp_min(eps)
    M = M / denom
    dead = (mx <= eps).expand_as(M)  # 全零角: 除以 eps 仍是 0 → 回退恒等门
    return torch.where(dead, torch.ones_like(M), M)


def tanh_offset_gate(logits: torch.Tensor, amplitude: float = GATE_AMPLITUDE) -> torch.Tensor:
    """次序 1: 网络原始输出 → 有界偏移场 M_raw = 1 + A·tanh(logits)."""
    return 1.0 + amplitude * torch.tanh(logits)


def content_gradient_density(c64: torch.Tensor) -> torch.Tensor:
    """c_64 (B,4,64,64) 梯度密度 ē: 16×16 per-sample min-max 归一到 [0,1] (低细节→0)."""
    dx = (c64[:, :, :, 1:] - c64[:, :, :, :-1]).abs()
    dy = (c64[:, :, 1:, :] - c64[:, :, :-1, :]).abs()
    g = torch.zeros(c64.shape[0], 1, *c64.shape[-2:], dtype=c64.dtype, device=c64.device)
    g[:, :, :, 1:] += dx.mean(dim=1, keepdim=True)
    g[:, :, 1:, :] += dy.mean(dim=1, keepdim=True)
    g16 = F.avg_pool2d(g, 4)  # (B,1,16,16)
    lo = g16.amin(dim=(2, 3), keepdim=True)
    hi = g16.amax(dim=(2, 3), keepdim=True)
    return (g16 - lo) / (hi - lo).clamp_min(1e-6)


def heuristic_gate(c64: torch.Tensor, kappa: float) -> torch.Tensor:
    """A1: M = 1 + κ(1−ē) → 统一 normalize_gate (低细节区给更大门控; 均值恒 1)."""
    return normalize_gate(1.0 + kappa * (1.0 - content_gradient_density(c64)))


def gate_input(c64: torch.Tensor, dz: torch.Tensor, chord: torch.Tensor) -> torch.Tensor:
    """GateNet 输入拼接: c_16(4) ⊕ Δz 下采样 16×16(4) ⊕ 风格弦 tile(128) → (B,136,16,16)."""
    c16 = F.avg_pool2d(c64.float(), 4)
    dz16 = F.avg_pool2d(dz.float(), 4)
    ch = chord.float()[:, :, None, None].expand(-1, -1, 16, 16)
    return torch.cat([c16, dz16, ch], dim=1)


class GateNet(nn.Module):
    """A2/A3 轻量门控网 (~0.30M ≤ 0.5M): ch64 ×3 ResBlock → 1×16×16, 末层零初始化.

    末层零初始化 ⇒ 初始 logits≡0 ⇒ M_raw≡1 ⇒ 归一后 M≡1 ⇒ z_out 与全局 γ 混合逐元素严格相等
    (A0 恒等热启动, 02 计划 A.2).
    """

    def __init__(self, chord_dim: int = 128, width: int = 64, blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4 + 4 + chord_dim, width, 3, padding=1),
            nn.GroupNorm(16, width),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(16, width),
                nn.Conv2d(width, width, 3, padding=d, dilation=d),
                nn.SiLU(inplace=True),
                nn.GroupNorm(16, width),
                nn.Conv2d(width, width, 3, padding=1),
            )
            for d in (1, 2, 2)
        ][:blocks])
        self.out_conv = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, c64: torch.Tensor, dz: torch.Tensor, chord: torch.Tensor) -> torch.Tensor:
        """返回归一化后的 16×16 门控场 M (B,1,16,16)."""
        h = self.stem(gate_input(c64, dz, chord))
        for blk in self.blocks:
            h = h + blk(h)
        return normalize_gate(tanh_offset_gate(self.out_conv(h)))


def upsample_gate(M16: torch.Tensor, size: int = 64) -> torch.Tensor:
    """16×16 → 64×64 双线性上采样 + 二次归一 (双线性不精确保均值; clamp 后必须再次 mean-norm)."""
    M64 = F.interpolate(M16.float(), size=(size, size), mode="bilinear", align_corners=False)
    return normalize_gate(M64)


def apply_gate(z_p1: torch.Tensor, dz: torch.Tensor, M16: torch.Tensor, gamma) -> torch.Tensor:
    """z_out = z_P1 + γ·M⊙Δz; gamma 可为标量或 per-task (B,) 张量 (能量匹配模式)."""
    M64 = upsample_gate(M16, size=z_p1.shape[-1])
    g = torch.as_tensor(gamma, dtype=z_p1.dtype, device=z_p1.device)
    return z_p1 + g.view(-1, 1, 1, 1) * M64 * dz


def edit_energy(z_out: torch.Tensor, z_base: torch.Tensor, dz: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """逐任务 edit_energy = ||z_out − z_base||₂ / ||Δz||₂ ≡ ||γ·M⊙Δz||₂/||Δz||₂ (A0 恒等于 γ)."""
    num = (z_out.detach().float() - z_base.detach().float()).flatten(1).norm(dim=1)
    den = dz.detach().float().flatten(1).norm(dim=1).clamp_min(eps)
    return num / den


def energy_matched_gamma(M16: torch.Tensor, dz: torch.Tensor, target_energy: float,
                         eps: float = 1e-8) -> torch.Tensor:
    """per-task 能量匹配 γ*_i = target / m_rms_i, m_rms_i = ||M⊙Δz_i||/||Δz_i||.

    使每个任务在能量匹配 γ* 下的 edit_energy 精确等于 target (A0 的 E ≡ 名义 γ).
    """
    M64 = upsample_gate(M16, size=dz.shape[-1])
    m_rms = (M64 * dz.detach().float()).flatten(1).norm(dim=1) / \
        dz.detach().float().flatten(1).norm(dim=1).clamp_min(eps)
    return target_energy / m_rms.clamp_min(eps)


def tv_loss(M: torch.Tensor) -> torch.Tensor:
    """A3: TV(M) = mean(|Δx M| + |Δy M|) — 抑制门控场高频区域边界伪影."""
    dx = (M[:, :, :, 1:] - M[:, :, :, :-1]).abs().mean()
    dy = (M[:, :, 1:, :] - M[:, :, :-1, :]).abs().mean()
    return dx + dy


def edge_continuity_loss(M: torch.Tensor, c64: torch.Tensor,
                         quantile: float = EDGE_QUANTILE) -> torch.Tensor:
    """A3: 高内容边缘两侧门控连续性惩罚 — dilated 边缘掩码内的 |∇M|.

    语义边界两侧门控若跳变, 解码后出现区域拼图伪影 (02 计划 A.2 视觉审计项的损失侧对齐).
    """
    e16 = content_gradient_density(c64).detach()
    thr = e16.flatten(1).quantile(quantile).view(-1, 1, 1, 1)
    mask = (e16 >= thr).float()
    mask = F.max_pool2d(mask, 3, stride=1, padding=1)  # 3×3 膨胀 → 覆盖边缘两侧
    dx = (M[:, :, :, 1:] - M[:, :, :, :-1]).abs()
    dy = (M[:, :, 1:, :] - M[:, :, :-1, :]).abs()
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return (mask * (dx + dy)).sum() / mask.sum().clamp_min(1.0)


# ============================================================================
# 2) TeacherSpec: 教师定义固定铁律 (训练与评测共用, 写入结果 JSON)
# ============================================================================

_SPEC_KEYS = ("d6_ckpt", "anchor", "solver", "branches", "gate", "p1_ckpt", "p1_lambda")


def _norm_separators(obj):
    """路径分隔符归一 (Windows 缓存 'runs\\x' 与 Linux 'runs/x' 必须判等)."""
    if isinstance(obj, str):
        return obj.replace("\\", "/")
    if isinstance(obj, dict):
        return {k: _norm_separators(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_norm_separators(v) for v in obj]
    return obj


@dataclass(frozen=True)
class TeacherSpec:
    """教师三元组铁律 (agent2.md §二第 2 条 / precompute_train_latents.TEACHER_SPEC).

    训练侧 (z_d6_tr 生成) 与评测侧 (z_d6_750 再生) 必须同一 (ckpt, anchor, solver, branches, gate).
    """

    d6_ckpt: str
    anchor: str
    solver: dict
    branches: str
    gate: str
    p1_ckpt: str
    p1_lambda: float

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in _SPEC_KEYS}
        d["solver"] = dict(self.solver)
        return _norm_separators(d)

    @classmethod
    def from_dict(cls, d: dict) -> "TeacherSpec":
        missing = [k for k in _SPEC_KEYS if k not in d]
        if missing:
            raise ValueError(f"TeacherSpec missing keys: {missing}")
        extra = [k for k in d if k not in _SPEC_KEYS]
        if extra:
            raise ValueError(f"TeacherSpec unexpected keys (教师定义变更须显式同步本 dataclass): {extra}")
        return cls(**{k: d[k] for k in _SPEC_KEYS})


# 唯一权威教师定义: 逐字段镜像 precompute 缓存 (单一事实来源, 禁止本地另写一份数值)
CANONICAL_TEACHER_SPEC = TeacherSpec.from_dict(_PRECOMPUTED_TEACHER_SPEC)


def as_teacher_spec(spec) -> TeacherSpec:
    return spec if isinstance(spec, TeacherSpec) else TeacherSpec.from_dict(spec)


def assert_same_teacher(a, b, context: str = "") -> None:
    """训练/评测/缓存三方教师定义一致性断言; 不一致立即 AssertionError (逐键报告)."""
    da, db = as_teacher_spec(a).to_dict(), as_teacher_spec(b).to_dict()
    if da != db:
        diff = [k for k in _SPEC_KEYS if da[k] != db[k]]
        raise AssertionError(
            f"teacher spec mismatch{f' [{context}]' if context else ''}; keys={diff}; "
            f"a={ {k: da[k] for k in diff} }, b={ {k: db[k] for k in diff} }"
        )


def load_teacher_cache(cache_path) -> dict:
    """消费 runs/v25/cache/train_latents_4400.pt 并强制 teacher_spec == CANONICAL."""
    import torch as _t
    payload = _t.load(cache_path, map_location="cpu", weights_only=False)
    for k in ("z0_tr", "z_p1_tr", "z_d6_tr", "sid_tr", "tid_tr", "teacher_spec"):
        if k not in payload:
            raise ValueError(f"teacher cache {cache_path} missing key: {k}")
    assert_same_teacher(payload["teacher_spec"], CANONICAL_TEACHER_SPEC, context=str(cache_path))
    return payload


# ============================================================================
# 3) ConvNeXt 训练专用损失 (训练特征, 非评测裁判 —— "不直接优化裁判"铁律)
# ============================================================================

def _mid_cos_pair(f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
    return (1.0 - (f_a * f_b).sum(dim=1)).mean()


def _style_moment_pair(f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
    """逐通道 mean/std 矩匹配 (分布级风格损失; 空间不对齐 → 不约束内容布局)."""
    ma, sa = f_a.mean(dim=(2, 3)), f_a.std(dim=(2, 3))
    mb, sb = f_b.mean(dim=(2, 3)), f_b.std(dim=(2, 3))
    return F.mse_loss(ma, mb) + F.mse_loss(sa, sb)


def cnx_mid_content_loss(cnx, dec_grad, dec_ref, z_hat: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
    """可微 S10 解码 z_hat → ConvNeXt stage2(32²)/stage4(16²) 空间 cos vs 解码 z_ref (v23 C1 同式)."""
    rgb_hat = decode_s10(dec_grad, z_hat, batch=len(z_hat), dtype=torch.bfloat16,
                         enable_grad=True, output_cpu=False)
    with torch.no_grad():
        rgb_ref = decode_s10(dec_ref, z_ref, batch=len(z_ref), output_cpu=False)
        f32_r, f16_r = extract_convnext_mid_spatial(cnx, rgb_ref)
    f32_h, f16_h = extract_convnext_mid_spatial(cnx, rgb_hat)
    return 0.5 * _mid_cos_pair(f32_h, f32_r) + 0.5 * _mid_cos_pair(f16_h, f16_r)


def cnx_style_loss(cnx, dec_grad, dec_ref, z_out: torch.Tensor, z_style_ref: torch.Tensor) -> torch.Tensor:
    """L_cnx-style: dec(z_out) 与 dec(z_D6^tr) 的多尺度通道矩匹配 (风格分布对齐)."""
    rgb_out = decode_s10(dec_grad, z_out, batch=len(z_out), dtype=torch.bfloat16,
                         enable_grad=True, output_cpu=False)
    with torch.no_grad():
        rgb_ref = decode_s10(dec_ref, z_style_ref, batch=len(z_style_ref), output_cpu=False)
        f32_r, f16_r = extract_convnext_mid_spatial(cnx, rgb_ref)
    f32_o, f16_o = extract_convnext_mid_spatial(cnx, rgb_out)
    return 0.5 * _style_moment_pair(f32_o, f32_r) + 0.5 * _style_moment_pair(f16_o, f16_r)


@torch.no_grad()
def cnx_mid_cos_value(cnx, dec, z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """内容水位代理指标 (λ_c 标定用): dec(z_a) vs dec(z_b) 的中尺度空间余弦, 越高内容越保真."""
    rgb_a = decode_s10(dec, z_a, batch=len(z_a), output_cpu=False)
    rgb_b = decode_s10(dec, z_b, batch=len(z_b), output_cpu=False)
    f32_a, f16_a = extract_convnext_mid_spatial(cnx, rgb_a)
    f32_b, f16_b = extract_convnext_mid_spatial(cnx, rgb_b)
    return float(0.5 * (1.0 - _mid_cos_pair(f32_a, f32_b)) + 0.5 * (1.0 - _mid_cos_pair(f16_a, f16_b)))


# ============================================================================
# 4) 条件构建 (训练侧 / 评测侧共用)
# ============================================================================

@torch.no_grad()
def fiber_chords(bundle: dict, fib: torch.Tensor, tid: torch.Tensor, device: str) -> torch.Tensor:
    """风格弦 chord = bary[tid] − encode(fib) (precompute/routed_student 同配方)."""
    fnd = bundle["fnd"]
    enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    enc.load_state_dict(fnd["fiber_state"])
    u = enc.encode(fib.to(device).float())
    bary = enc.class_barycenters()
    chord = bary[tid.to(device)] - u
    del enc
    return chord


def arm_gate_field(arm: str, c64: torch.Tensor, dz: torch.Tensor, chord: torch.Tensor,
                   net: GateNet | None = None, kappa: float = 0.5) -> torch.Tensor:
    """四臂 → 16×16 门控场 M (B,1,16,16); A0 恒 1 / A1 启发式 / A2,A3 GateNet."""
    if arm == "A0_global":
        return torch.ones(c64.shape[0], 1, 16, 16, dtype=torch.float32, device=c64.device)
    if arm == "A1_heuristic":
        return heuristic_gate(c64, kappa)
    if arm in ("A2_learned", "A3_smooth"):
        if net is None:
            raise ValueError(f"arm {arm} requires a trained GateNet")
        return net(c64, dz, chord)
    raise ValueError(f"unknown mechanism-A arm: {arm}")


# ============================================================================
# 5) 训练 (4090 入口; 本任务交付不启动)
# ============================================================================

def _load_train_conditions(bundle: dict, cache: dict, device: str):
    """训练侧张量: cache 教师目标 + bundle 结构条件 + 风格弦 (一次构建, 各臂复用)."""
    tr = bundle["v18_cache"]["train"]
    device_t = torch.device(device)
    z_p1 = cache["z_p1_tr"].float()
    z_d6 = cache["z_d6_tr"].float()
    z0 = cache["z0_tr"].float()
    tid = cache["tid_tr"].long()
    c64 = tr["c_64"].float()
    chord = fiber_chords(bundle, bundle["fnd"]["tr_fib"].float(), tid, device).cpu()
    # 注: GateNet 输入 = c_16 ⊕ Δz16 ⊕ 风格弦 tile (02 计划 A.2), 不消费 d16
    return dict(z_p1=z_p1, z_d6=z_d6, z0=z0, tid=tid, c64=c64, chord=chord,
                n=len(z_p1), device_t=device_t)


def _gate_train_step(net, cond, gamma, lam_c, cnx, dec_g, dec_t, arm, decode_sub,
                     device_t, cpu_gen):
    """单 micro 步: 采样 → 门控前向 → cnx style + λ_c·mid 内容损失 (可微 S10)."""
    idx = torch.randint(0, cond["n"], (A_MICRO,), generator=cpu_gen)
    z_p1 = cond["z_p1"][idx].to(device_t)
    z_d6 = cond["z_d6"][idx].to(device_t)
    z0 = cond["z0"][idx].to(device_t)
    c64 = cond["c64"][idx].to(device_t)
    chord = cond["chord"][idx].to(device_t)
    dz = z_d6 - z_p1

    M16 = net(c64, dz, chord)
    z_out = apply_gate(z_p1, dz, M16, gamma)
    sub = slice(0, min(decode_sub, len(idx)))
    loss_st = cnx_style_loss(cnx, dec_g, dec_t, z_out[sub], z_d6[sub])
    loss_mid = cnx_mid_content_loss(cnx, dec_g, dec_t, z_out[sub], z0[sub])
    loss = loss_st + lam_c * loss_mid
    if arm == "A3_smooth":
        loss = loss + TV_WEIGHT * tv_loss(M16) + EDGE_WEIGHT * edge_continuity_loss(M16, c64)
    return loss, float(loss_st.detach()), float(loss_mid.detach())


def train_gate_arm(cond: dict, arm: str, gamma: float, lam_c: float, cnx, dec_g, dec_t,
                   device: str, updates: int = 2000, seed: int = A_SEED, tag: str | None = None,
                   save: bool = True):
    """只训 GateNet (冻结一切教师/基座); r2 工程规范: seed_all + cosine lr + micro/accum + EMA."""
    assert arm in ("A2_learned", "A3_smooth"), f"arm {arm} has no trainable params"
    seed_all(seed)
    device_t = cond["device_t"]
    net = GateNet().to(device_t)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=6e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    micro, accum = tier_micro_accum(A_EFFECTIVE, default_micro_24gb=A_MICRO,
                                    max_micro_12gb=A_MICRO, device=device)
    cpu_gen = torch.Generator().manual_seed(seed + 1)
    t0, st_v, mid_v = time.time(), 0.0, 0.0

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 6e-5 + 0.5 * (6e-4 - 6e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            loss, st_v_s, mid_v_s = _gate_train_step(net, cond, gamma, lam_c, cnx, dec_g,
                                                     dec_t, arm, decode_sub=8, device_t=device_t,
                                                     cpu_gen=cpu_gen)
            (loss / accum).backward()
            st_v += st_v_s / accum
            mid_v += mid_v_s / accum
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[MechA {arm}] {step:4d}/{updates} style={st_v/step:.5f} mid={mid_v/step:.5f} "
                  f"lr={lr:.2e} ({time.time()-t0:.0f}s)", flush=True)

    info = dict(arm=arm, gamma=gamma, lam_c=lam_c, updates=updates, n_params=net.n_params(),
                micro=micro, accum=accum, final_style=st_v / updates, final_mid=mid_v / updates,
                teacher_spec=CANONICAL_TEACHER_SPEC.to_dict())
    ckpt_p = None
    if save:
        out_dir = OUT / "mechanism_a"
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_p = out_dir / f"gate_{tag or arm}_g{int(round(gamma*100)):02d}.pt"
        torch.save({"ema": ema.state_dict(), "model": net.state_dict(), "info": info}, ckpt_p)
        info["ckpt_path"] = str(ckpt_p.relative_to(ROOT))
    return ema.eval(), info


@torch.no_grad()
def _probe_mid_cos(net, cond, probe_idx, gamma, cnx, dec, device_t) -> float:
    """标定探针: 固定子集上 A2 门控输出的中尺度内容余弦 (越低=改动越大)."""
    z_p1 = cond["z_p1"][probe_idx].to(device_t)
    z_d6 = cond["z_d6"][probe_idx].to(device_t)
    z0 = cond["z0"][probe_idx].to(device_t)
    c64 = cond["c64"][probe_idx].to(device_t)
    chord = cond["chord"][probe_idx].to(device_t)
    M16 = net(c64, z_d6 - z_p1, chord)
    z_out = apply_gate(z_p1, z_d6 - z_p1, M16, gamma)
    return cnx_mid_cos_value(cnx, dec, z_out, z0)


def calibrate_lambda_c(cond: dict, gamma: float, cnx, dec_g, dec_t, device: str,
                       candidates=(0.3, 0.6, 1.0), cal_updates: int = 300) -> tuple[float, dict]:
    """λ_c 校准 (02 计划 A.3: 先 300 updates 标定).

    内容水位目标 = A0 全局混合在同一 γ 的探针中尺度余弦 —— Round 1 该 γ 处 DINO-C≈0.73,
    故"匹配 A0 内容水位"即"DINO-C≈0.73 水位"的可执行操作化 (不循环引用评测裁判).
    """
    device_t = cond["device_t"]
    g = torch.Generator().manual_seed(CAL_SEED)
    probe_idx = torch.randperm(cond["n"], generator=g)[:64]
    # A0 无网络: 直接构造全局混合 z_p1 + γ·Δz 作为内容水位锚
    z_p1 = cond["z_p1"][probe_idx].to(device_t)
    z_d6 = cond["z_d6"][probe_idx].to(device_t)
    z0 = cond["z0"][probe_idx].to(device_t)
    target_cos = cnx_mid_cos_value(cnx, dec_t, z_p1 + gamma * (z_d6 - z_p1), z0)
    print(f"[MechA cal] A0@γ={gamma} probe mid-cos={target_cos:.5f} (content target)", flush=True)

    measured = {}
    for lam in candidates:
        net, _ = train_gate_arm(cond, "A2_learned", gamma, lam, cnx, dec_g, dec_t, device,
                                updates=cal_updates, save=False)
        cos = _probe_mid_cos(net, cond, probe_idx, gamma, cnx, dec_t, device_t)
        measured[lam] = cos
        print(f"[MechA cal] λ_c={lam}: probe mid-cos={cos:.5f} (Δ={cos-target_cos:+.5f})", flush=True)
    picked = min(candidates, key=lambda lam: abs(measured[lam] - target_cos))
    info = dict(gamma=gamma, target_mid_cos=round(target_cos, 6),
                measured={str(k): round(v, 6) for k, v in measured.items()}, picked=picked,
                candidates=list(candidates), cal_updates=cal_updates)
    save_json(OUT / "mechanism_a" / "lambda_c_calibration.json",
              {"updated_utc": utc(), **info})
    return float(picked), info


# ============================================================================
# 6) 评测: 教师口径 test750 latent 构建 + 四臂门控应用 + 能量记录
# ============================================================================

@torch.no_grad()
def build_teacher_latents_750(bundle: dict, bank: dict, device: str,
                              spec: TeacherSpec | None = None) -> dict:
    """评测侧教师 latent: z_p1_750 (PRISM-1 λ=1.0) + z_d6_750 (spec 三元组逐位再生).

    注意: V24 缓存的 z_d6 是 full-CFG 口径, 不满足教师铁律 —— 此处按 spec.solver/branches/gate
    经 run_core_flow 重新生成; z_p1_750 用 compute_prism1_latents (与教师侧同配方).
    """
    spec = spec or CANONICAL_TEACHER_SPEC
    from exp.v22.t4 import build_or_load_t4_latents_750

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    z_p1_750 = compute_prism1_latents(bundle, tasks, z_src_750, d16_750,
                                      lam=float(spec.p1_lambda), device=device)
    z_d6_750, tau_stats = run_core_flow(
        bundle, bank, V20_CKPT, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750,
        branches=spec.branches, gate_mode=spec.gate, solver=dict(spec.solver), device=device,
    )
    sid = torch.tensor([t["source_style_id"] for t in tasks])
    tid = torch.tensor([t["target_style_id"] for t in tasks])
    same = sid == tid
    chords = fiber_chords(bundle, bundle["fnd"]["test750_fib"].float(), tid, device).cpu()
    chords[same] = 0.0  # same 任务走网络外 bypass, 弦置零保持门控输入良性
    dec = load_s10_decoder(device)
    return dict(tasks=tasks, z_src=z_src_750.float(), z_p1=z_p1_750, z_d6=z_d6_750,
                c64=bundle["test_struct"]["c_64"].float(), d16=d16_750.float(),
                chords=chords, sid=sid, tid=tid, same=same, dec=dec, tau_stats=tau_stats)


@torch.no_grad()
def eval_mechanism_a_arm(ctx750: dict, arm: str, gamma: float, device: str, net=None,
                         kappa: float | None = None, energy_match: bool = False,
                         measure: bool = True, batch: int = 50) -> dict:
    """单臂评测: 门控应用 + 逐任务 E 记录 + (可选) 完整 test750 四裁判管线.

    A2/A3 同时产出名义 γ 与能量匹配两个记录 (agent2.md §二: 比较相近 E 而非相同名义 γ).
    """
    z_p1, z_d6, z_src = ctx750["z_p1"], ctx750["z_d6"], ctx750["z_src"]
    dz = (z_d6 - z_p1).float()
    same = ctx750["same"]
    out = torch.zeros_like(z_p1)
    e_rec = []
    for st in range(0, len(z_p1), batch):
        sl = slice(st, min(st + batch, len(z_p1)))
        c64 = ctx750["c64"][sl].to(device)
        d16 = ctx750["d16"][sl].to(device)
        chord = ctx750["chords"][sl].to(device)
        M16 = arm_gate_field(arm, c64, dz[sl].to(device), chord, net=net, kappa=kappa)
        g = energy_matched_gamma(M16, dz[sl].to(device), gamma) if energy_match else gamma
        z_out = apply_gate(z_p1[sl].to(device), dz[sl].to(device), M16, g)
        z_out[torch.as_tensor(same[sl]).to(device)] = z_src[sl].to(device)  # bypass 网络之外
        out[sl] = z_out.cpu()
        e_rec.append(edit_energy(z_out, z_p1[sl].to(device), dz[sl].to(device)).cpu())
    e = torch.cat(e_rec)[~same]
    e_style = {s[:3]: round(float(e[ctx750["tid"][~same] == i].mean()), 5)
               for i, s in enumerate(STYLES)}
    tag = arm if kappa is None else f"{arm}_k{int(round(kappa*100)):02d}"
    tag = f"{tag}_em" if energy_match else tag
    rec = dict(
        arm=tag, nominal_gamma=gamma, kappa=(kappa if arm == "A1_heuristic" else None),
        energy_matched=energy_match,
        edit_energy_mean=round(float(e.mean()), 5), edit_energy_p50=round(float(e.median()), 5),
        edit_energy_max=round(float(e.max()), 5), edit_energy_per_style=e_style,
        tau_stats=ctx750["tau_stats"], teacher_spec=CANONICAL_TEACHER_SPEC.to_dict(),
    )
    if measure:
        m = eval_latent_v25(f"MechA_{tag}_g{int(round(gamma*100)):02d}", out, ctx750["tasks"],
                            ctx750["dec"], device=device)
        rec.update({k: v for k, v in m.items() if k != "arm"})
    return rec


# ============================================================================
# 7) main (4090 完整入口; 本任务只交付, 不启动 GPU)
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", help="A0_global | A1_heuristic | A2_learned | A3_smooth | all")
    ap.add_argument("--gamma", type=float, default=TRAIN_GAMMA_DEFAULT)
    ap.add_argument("--kappa", default="0.5,1.0", help="A1 κ 网格 (逗号分隔)")
    ap.add_argument("--lam-c", type=float, default=-1.0, help="<0 → 300-update 自动标定")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--cal-updates", type=int, default=300)
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-eval", action="store_true", help="只训练不评测 (调试用)")
    args = ap.parse_args()

    sanitize_cuda_env()
    cache_p = OUT / "cache/train_latents_4400.pt"
    if not cache_p.exists():
        raise FileNotFoundError(f"{cache_p} 不存在 — 先运行 python -m exp.v25.precompute_train_latents")
    cache = load_teacher_cache(cache_p)
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    cond = _load_train_conditions(bundle, cache, args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    dec = load_s10_decoder(args.device)
    dec_g = copy.deepcopy(dec).to(device=args.device, dtype=torch.bfloat16).eval()
    for p in dec_g.parameters():
        p.requires_grad_(False)

    arms = ("A0_global", "A1_heuristic", "A2_learned", "A3_smooth") \
        if args.arm == "all" else (args.arm,)
    learned = [a for a in arms if a in ("A2_learned", "A3_smooth")]
    lam_c = args.lam_c
    if learned and lam_c < 0:
        lam_c, cal_info = calibrate_lambda_c(cond, args.gamma, cnx, dec_g, dec, args.device,
                                             cal_updates=args.cal_updates)
        print(f"[MechA] calibrated λ_c={lam_c} ({cal_info})", flush=True)

    nets = {}
    out_json = OUT / "mechanism_a" / "mechanism_a.json"
    results = []
    for arm in arms:
        if arm in ("A2_learned", "A3_smooth"):
            nets[arm], info = train_gate_arm(cond, arm, args.gamma, lam_c, cnx, dec_g, dec,
                                             args.device, updates=args.updates)
            print(f"[MechA] trained {arm}: {info}", flush=True)
    if not args.skip_eval:
        ctx750 = build_teacher_latents_750(bundle, bank, args.device)  # 一次构建, 各臂复用
    for arm in arms:
        if args.skip_eval:
            continue
        if arm == "A1_heuristic":
            kappas = [float(k) for k in str(args.kappa).split(",") if k.strip()]
        else:
            kappas = [None]  # 非 A1 臂不携带 κ (臂名不加 _k 后缀)
        for kappa in kappas:
            results.append(eval_mechanism_a_arm(ctx750, arm, args.gamma, args.device,
                                                net=nets.get(arm), kappa=kappa,
                                                energy_match=False))
            if arm in ("A2_learned", "A3_smooth"):
                results.append(eval_mechanism_a_arm(ctx750, arm, args.gamma, args.device,
                                                    net=nets[arm], kappa=kappa,
                                                    energy_match=True))
            save_json(out_json, {"updated_utc": utc(), "gamma": args.gamma,
                                 "lam_c": lam_c, "teacher_spec": CANONICAL_TEACHER_SPEC.to_dict(),
                                 "results": results})
    print(f"\n=== V25 机制 A 完成: {len(results)} 条记录 -> {out_json} ===", flush=True)
    for r in results:
        m = {k: r[k] for k in ("dino_c", "dino_s", "lpips") if k in r}
        print(f"{r['arm']:22s} γ={r['nominal_gamma']:.2f} E={r['edit_energy_mean']:.4f} {m}", flush=True)


if __name__ == "__main__":
    main()
