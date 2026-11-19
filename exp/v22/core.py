"""P0 统一主模型接口：可替换组件 + 与 V20 求解器逐位等价的参考实现。

组件: build_style_condition / build_content_condition / make_initial_state /
      predict_guided_velocity / integrate_flow / decode_output

C组门控解耦语义: 原始 per-sample g_raw 恒被计算（sigmoid 或 ECDF），
GateConfig.drive_{tau,layout,guidance} 决定每个消费方使用 g_raw 还是 fixed_g。
original = 三者全由 g_raw 驱动; fixed = 三者全用 fixed_g。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import EVAL_SEED, autocast
from exp.v20.slf_flow import SLFUNet
from exp.v20.submode_patchswd import match_scene_submode

GATE_SLOPE = 550.0
D5_SOLVER = dict(
    tau_min=0.28, tau_hi=0.66, steps=10,
    ws_fig=4.2, wsp_fig_late=2.4, wsp_abs=5.2,
    wc_fig=1.85, wc_abs=0.80, lscale_abs=0.34, rescale_phi=0.48,
)
D6_SOLVER = dict(
    tau_min=0.40, tau_hi=0.74, steps=8,
    ws_fig=4.4, wsp_fig_late=1.8, wsp_abs=4.8,
    wc_fig=1.95, wc_abs=1.00, lscale_abs=0.44, rescale_phi=0.48,
)
D2_SOLVER = dict(
    tau_min=0.00, tau_hi=0.44, steps=16,
    ws_fig=3.8, wsp_fig_late=4.8, wsp_abs=5.4,
    wc_fig=1.60, wc_abs=0.60, lscale_abs=0.28, rescale_phi=0.48,
)


@dataclass
class GateConfig:
    """C组: 密度门控解耦。mode: original | fixed | quantile。"""
    mode: str = "original"      # original: sigmoid((dsd-mid)*550); quantile: train sub_dsd ECDF; fixed: 常数
    fixed_g: float = 0.5        # fixed / 解耦后的固定分量（train 中位数校准）
    drive_tau: bool = True
    drive_layout: bool = True
    drive_guidance: bool = True
    quantile_ref: torch.Tensor | None = field(default=None, repr=False)  # C5: train sub_dsd 样本


@dataclass
class CFGConfig:
    """B组: 分支组合。branches: full | sc_only | cs_j | no_wp | no_wc。"""
    branches: str = "full"
    rescale_phi: float = 0.48


def effective_gate_components(dsd_ref: torch.Tensor, gate: GateConfig, mid_dsd: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (g_tau, g_layout, g_guid, g_rescale) —— 每个消费方实际使用的 per-sample g。"""
    if gate.mode == "original":
        g_raw = torch.sigmoid((dsd_ref - float(mid_dsd)) * GATE_SLOPE)
    elif gate.mode == "quantile":
        ref = gate.quantile_ref.to(dsd_ref.device).float().flatten().sort().values
        pos = torch.searchsorted(ref, dsd_ref.float().contiguous())
        g_raw = (pos.float() / max(1, len(ref))).clamp(0.02, 0.98)
    elif gate.mode == "fixed":
        g_raw = torch.full_like(dsd_ref, float(gate.fixed_g))
    else:
        raise ValueError(gate.mode)
    fixed = torch.full_like(dsd_ref, float(gate.fixed_g))
    g_tau = g_raw if gate.drive_tau else fixed
    g_layout = g_raw if gate.drive_layout else fixed
    g_guid = g_raw if gate.drive_guidance else fixed
    g_rescale = g_guid
    return g_tau, g_layout, g_guid, g_rescale


def build_content_condition(y16: torch.Tensor, d16: torch.Tensor, g_layout: torch.Tensor) -> dict:
    """16×16 语义 + 亮度条件；g_layout 驱动 coarse8/16 混合（逐位复刻 V20）。"""
    B = y16.shape[0]
    c8_w = (1.0 - g_layout).clamp(0.0, 0.85).view(B, 1, 1, 1)
    d8 = F.avg_pool2d(d16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    d8 = d8 - d8.mean(dim=(2, 3), keepdim=True)
    y8 = F.avg_pool2d(y16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    d_in = (1.0 - c8_w) * d16 + c8_w * d8
    y_in = (1.0 - c8_w) * y16 + c8_w * y8
    return {"y_in": y_in, "d_in": d_in}


def make_initial_state(eps: torch.Tensor, z_anchor: torch.Tensor | None, tau_v: torch.Tensor) -> torch.Tensor:
    """z(tau0) = (1-tau0)*noise + tau0*anchor; anchor=None 时强制 tau0=0（纯噪声 A0）。"""
    if z_anchor is None:
        return eps.clone()
    B = eps.shape[0]
    return (1.0 - tau_v.view(B, 1, 1, 1)) * eps + tau_v.view(B, 1, 1, 1) * z_anchor


def _guidance_curves(g_guid: torch.Tensor, solver: dict, prog: float):
    """逐位复刻 V20: wc/ws_j/ws_p 的 fig/abs 双点插值 + 后期线性爬升 gl。"""
    gf = g_guid.view(-1, 1, 1, 1)
    gl = min(1.0, max(0.0, (prog - 0.36) / 0.32))
    wc = gf * (solver["wc_fig"] - 0.45 * gl) + (1.0 - gf) * solver["wc_abs"]
    ws_j = gf * (solver["ws_fig"] - 1.1 * gl) + (1.0 - gf) * 1.4
    ws_p = gf * (gl * solver["wsp_fig_late"]) + (1.0 - gf) * solver["wsp_abs"]
    return wc, ws_j, ws_p


def predict_guided_velocity(
    model,
    z_cur: torch.Tensor,
    tv: torch.Tensor,
    content: dict,
    tid: torch.Tensor,
    u_style: torch.Tensor,
    lscale: torch.Tensor,
    g_guid: torch.Tensor,
    cfg: CFGConfig,
    solver: dict,
    prog: float,
    y32: torch.Tensor | None = None,
    is_s32: bool = False,
) -> torch.Tensor:
    """分支组合（批量前向逐位复刻 V20 的 repeat/cat 顺序）。y32 仅 S32 架构使用。"""
    B = z_cur.shape[0]
    dev = z_cur.device
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls0 = torch.zeros(B, device=dev)
    y32_kw = {"y32_struct": y32} if is_s32 else {}

    if cfg.branches == "sc_only":
        with autocast(str(dev)):
            return model(z_cur, content["y_in"], content["d_in"], tid, u_style, tv,
                         drop_style=f0, drop_content=f0, layout_scale=lscale, **y32_kw).float()

    if cfg.branches == "cs_j":
        z2 = torch.cat([z_cur, z_cur], 0)
        y2 = content["y_in"].repeat(2, 1, 1, 1)
        d2 = content["d_in"].repeat(2, 1, 1, 1)
        s2 = tid.repeat(2)
        u2 = u_style.repeat(2, 1)
        t2 = tv.repeat(2)
        ds2 = torch.cat([f0, f1], 0)
        dc2 = torch.cat([f0, f0], 0)
        l2 = torch.cat([lscale, lscale], 0)  # V23-P0-A 修复: v_0c 必须保留内容布局 lscale，不得清零
        y32_2 = y32.repeat(2, 1, 1, 1) if (is_s32 and y32 is not None) else None
        with autocast(str(dev)):
            v_sc, v_0c = model(z2, y2, d2, s2, u2, t2, drop_style=ds2, drop_content=dc2, layout_scale=l2,
                               **({"y32_struct": y32_2} if is_s32 else {})).float().chunk(2, 0)
        _, ws_j, _ = _guidance_curves(g_guid, solver, prog)
        return v_0c + ws_j * (v_sc - v_0c)

    if cfg.branches in ("full", "no_wp", "no_wc"):
        z4 = z_cur.repeat(4, 1, 1, 1)
        y4 = content["y_in"].repeat(4, 1, 1, 1)
        d4 = content["d_in"].repeat(4, 1, 1, 1)
        s4 = tid.repeat(4)
        u4 = u_style.repeat(4, 1)
        t4 = tv.repeat(4)
        ds = torch.cat([f0, f0, f1, f1], 0)
        dc = torch.cat([f0, f1, f0, f1], 0)
        lsc4 = torch.cat([lscale, ls0, lscale, ls0], 0)
        y32_4 = y32.repeat(4, 1, 1, 1) if (is_s32 and y32 is not None) else None
        with autocast(str(dev)):
            v_sc, v_s0, v_0c, v_00 = model(
                z4, y4, d4, s4, u4, t4, drop_style=ds, drop_content=dc, layout_scale=lsc4,
                **({"y32_struct": y32_4} if is_s32 else {}),
            ).float().chunk(4, 0)
        wc, ws_j, ws_p = _guidance_curves(g_guid, solver, prog)
        if cfg.branches == "no_wp":
            ws_p = torch.zeros_like(ws_p)
        if cfg.branches == "no_wc":
            wc = torch.zeros_like(wc)
        vr = v_00 + wc * (v_0c - v_00) + ws_j * (v_sc - v_0c) + ws_p * (v_s0 - v_00)
        if cfg.rescale_phi > 0.0:
            gf4 = g_guid.view(B, 1, 1, 1)
            s_ref = (gf4 * v_sc + (1.0 - gf4) * v_s0).flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
            s_raw = vr.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
            vr = cfg.rescale_phi * vr * (s_ref / s_raw) + (1.0 - cfg.rescale_phi) * vr
        return vr

    raise ValueError(f"unknown branches: {cfg.branches}")


def rk2_integrate(vel_fn, z0: torch.Tensor, tau_v: torch.Tensor, steps: int) -> torch.Tensor:
    """共享 RK2 中点积分循环（per-sample tau_v）；vel_fn(z, prog) -> v。与 V20 循环逐位一致。

    prog 取值必须与 V20 完全一致: v1 用 i/steps, v2 用 (i+0.5)/steps（浮点不等价于 i/steps+0.5/steps）。
    """
    B = z0.shape[0]
    dt = (1.0 - tau_v).view(B, 1, 1, 1) / float(steps)
    z = z0
    for i in range(int(steps)):
        prog1 = i / float(steps)
        v1 = vel_fn(z, prog1)
        zm = z + 0.5 * dt * v1
        v2 = vel_fn(zm, (i + 0.5) / float(steps))
        z = z + dt * v2
    return z


class PrismCore:
    """绑定 checkpoint + bank 的统一推理入口（一个消融只改变一个组件）。"""

    def __init__(self, bundle: dict | None = None, core_bank: dict | None = None,
                 ckpt_path: str | Path | None = None, device: str = "cuda",
                 model: SLFUNet | None = None, arch: str = "base"):
        self.device = device
        self.bank = core_bank or {}
        self.bundle = bundle
        self.arch = arch
        if model is not None:
            self.model = model.to(device).eval()
        elif arch == "s32":
            from .arch_s32 import SLFUNetS32
            self.model = SLFUNetS32(base_dim=144, u_dim=128).to(device).eval()
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ck["ema"], strict=True)
        else:
            self.model = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ck["ema"], strict=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.ckpt_sha: str | None = None
        self.fiber_enc = None  # 惰性: 仅 core1 / random / self-reference 需要

    # ---------------- style condition ----------------
    def _ensure_fiber(self):
        if self.fiber_enc is None:
            from exp.v19.fiber_duality import StyleFiberEncoder
            fnd = self.bundle["fnd"]
            enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(self.device).eval()
            enc.load_state_dict(fnd["fiber_state"])
            for p in enc.parameters():
                p.requires_grad_(False)
            self.fiber_enc = enc
        return self.fiber_enc

    def build_style_condition(
        self, d16: torch.Tensor, tid: torch.Tensor, mode: str = "core8",
        task_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (u (B,128), dsd_ref (B,))。mode: core8 | core4 | core1 | random。"""
        dev = d16.device
        bank = self.bank
        if mode == "core8":
            _, u = match_scene_submode(d16, tid, bank)
            sc_src = F.normalize(F.avg_pool2d(d16.float(), 4).flatten(1), dim=-1)
            best_c = torch.einsum("bd,bkd->bk", sc_src, bank["sub_scene"].to(dev).float()[tid]).argmax(dim=1)
            dsd = bank["sub_dsd"].to(dev).float()[tid, best_c]
            return u, dsd
        if mode == "core4":
            u4c, sc4c, dsd4c = self._merged_protos(k_keep=4)
            return self._match_protos(d16, tid, u4c, sc4c, dsd4c)
        if mode == "core1":
            enc = self._ensure_fiber()
            u = enc.class_barycenters()[tid]
            dsd = bank["sub_dsd"].to(dev).float()[tid].mean(dim=1)
            return u, dsd
        if mode == "random":
            enc = self._ensure_fiber()
            pools = [ci.detach().cpu() for ci in bank["core_idx_per_style"]]
            u_exact = bank["u_exact_all"]
            us = []
            for j, t in enumerate(tid.detach().cpu().tolist()):
                seed = int(EVAL_SEED) + (int(task_index[j]) if task_index is not None else j)
                g = torch.Generator().manual_seed(seed)
                pool = pools[int(t)]
                pick = int(pool[torch.randint(0, len(pool), (1,), generator=g)])
                us.append(u_exact[pick])
            u = torch.stack(us).to(dev).float()
            dsd = bank["sub_dsd"].to(dev).float()[tid].mean(dim=1)
            return u, dsd
        raise ValueError(mode)

    def _merged_protos(self, k_keep: int = 4):
        from exp.v20.submode_patchswd import _kmeans_unit
        bank = self.bank
        sub_u = bank["sub_u"].float()       # (5, 8, 128)
        sub_sc = bank["sub_scene"].float()  # (5, 8, D)
        sub_dsd = bank["sub_dsd"].float()   # (5, 8)
        Us, Ss, Ds = [], [], []
        for s in range(5):
            centers, labels = _kmeans_unit(F.normalize(sub_u[s], dim=-1), k=k_keep, iters=30, seed=900 + s)
            u_k, sc_k, dsd_k = [], [], []
            for c in range(k_keep):
                m = (labels == c)
                if m.sum() == 0:
                    m = torch.ones(len(labels), dtype=torch.bool)
                u_k.append(F.normalize(sub_u[s][m].mean(0), dim=-1))
                sc_k.append(F.normalize(sub_sc[s][m].mean(0), dim=-1))
                dsd_k.append(sub_dsd[s][m].mean())
            Us.append(torch.stack(u_k))
            Ss.append(torch.stack(sc_k))
            Ds.append(torch.stack(dsd_k))
        return torch.stack(Us), torch.stack(Ss), torch.stack(Ds)

    def _match_protos(self, d16, tid, u_p, sc_p, dsd_p):
        dev = d16.device
        sc_src = F.normalize(F.avg_pool2d(d16.float(), 4).flatten(1), dim=-1)
        sim = torch.einsum("bd,bkd->bk", sc_src, sc_p.to(dev)[tid])
        best = sim.argmax(dim=1)
        return u_p.to(dev)[tid, best], dsd_p.to(dev)[tid, best]

    def self_reference_u(self, fib_750: torch.Tensor) -> torch.Tensor:
        enc = self._ensure_fiber()
        return enc.encode(fib_750.to(self.device).float())

    # ---------------- solver ----------------
    def integrate_flow(
        self,
        eps: torch.Tensor,
        z_anchor: torch.Tensor | None,
        y16: torch.Tensor,
        d16: torch.Tensor,
        tid: torch.Tensor,
        *,
        u_style: torch.Tensor | None = None,
        dsd_ref: torch.Tensor | None = None,
        style_mode: str = "core8",
        gate: GateConfig | None = None,
        cfg: CFGConfig | None = None,
        solver: dict | None = None,
        task_index: torch.Tensor | None = None,
        y32: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """gate=original / branches=full / style=None 时与 exp.v20.integrate_slf_core_batch 逐位等价。"""
        solver = dict(D5_SOLVER if solver is None else solver)
        gate = gate or GateConfig()
        cfg = cfg or CFGConfig()
        dev = self.device
        B = eps.shape[0]

        if u_style is None or dsd_ref is None:
            u_style, dsd_ref = self.build_style_condition(
                d16.to(dev).float(), tid, mode=style_mode, task_index=task_index,
            )
        u_style = u_style.to(dev).float()
        dsd_ref = dsd_ref.to(dev).float()

        # C5: 分位数门控缺训练分布时，自动取 Core bank 的 train sub_dsd（5×K 核心原型 token 方差）
        if gate.mode == "quantile" and gate.quantile_ref is None:
            gate.quantile_ref = self.bank["sub_dsd"].detach().float().flatten().clone()

        g_tau, g_layout, g_guid, g_rescale = effective_gate_components(
            dsd_ref, gate, float(self.bank["mid_dsd"]),
        )

        lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
        content = build_content_condition(y16.to(dev).float(), d16.to(dev).float(), g_layout)

        tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
        if z_anchor is None:
            tau_v = torch.zeros_like(tau_v)  # A0: 纯噪声起点，时间标签同步归零
        z = make_initial_state(eps.to(dev).float(), None if z_anchor is None else z_anchor.to(dev).float(), tau_v)

        steps = int(solver["steps"])
        y32_dev = y32.to(dev).float() if (self.arch == "s32" and y32 is not None) else None
        is_s32 = self.arch == "s32"
        vel_fn = lambda zz, prog: predict_guided_velocity(
            self.model, zz, tau_v + prog * (1.0 - tau_v), content, tid, u_style,
            lscale, g_guid, cfg, solver, prog, y32=y32_dev, is_s32=is_s32,
        )
        z = rk2_integrate(vel_fn, z, tau_v, steps)
        stats = {
            "g_mean": float(g_tau.mean()),
            "tau_mean": float(tau_v.mean()),
            "lscale_mean": float(lscale.mean()),
        }
        return z, stats

    # ---------------- decode ----------------
    def decode_output(self, z: torch.Tensor, dec, batch: int = 16) -> torch.Tensor:
        from src.utils import decode_s10
        return decode_s10(dec, z, batch=batch).clamp(0.0, 1.0)


def latent_change_rate(z_out: torch.Tensor, z_src: torch.Tensor) -> float:
    dz = (z_out.detach().float().cpu() - z_src.detach().float().cpu()).flatten(1)
    return float((dz.norm(dim=1) / z_src.detach().float().cpu().flatten(1).norm(dim=1).clamp_min(1e-6)).mean())


def coarse_luminance_pullback(
    z: torch.Tensor,
    z_anchor: torch.Tensor,
    w_luma: torch.Tensor,
    b_luma: torch.Tensor,
    beta: float = 0.2,
    pool: int = 8,
) -> torch.Tensor:
    """P3-N2 粗结构信任域（无训练、推理期一次修正）：
    把 z 的粗尺度（pool×pool 块）亮度场向锚点拉回 beta 比例；沿潜亮度方向 ŵ_Y 作用，零色度泄漏。
    """
    dev = z.device
    w = w_luma.to(dev).float().view(1, 4, 1, 1)
    b = b_luma.to(dev).float()
    w_norm = float(w_luma.to(dev).float().norm())
    yz = ((z.float() * w).sum(dim=1, keepdim=True) + b).clamp(0.0, 1.0)
    ya = ((z_anchor.float() * w).sum(dim=1, keepdim=True) + b).clamp(0.0, 1.0)
    lpz = F.avg_pool2d(yz, pool, stride=pool)
    lpa = F.avg_pool2d(ya, pool, stride=pool)
    delta_y = beta * (lpa - lpz)                       # (B,1,H/p,W/p) 粗亮度差
    delta_y = F.interpolate(delta_y, size=yz.shape[-2:], mode="bilinear", align_corners=False)
    # 沿 ŵ_Y 方向施加: Y(z + c·ŵ) = Y(z) + c·||w|| → c = delta_y / ||w||
    unit_w = (w / max(w_norm, 1e-6)).view(1, 4, 1, 1)
    return (z.float() + delta_y / max(w_norm, 1e-6) * unit_w).to(z.dtype)
