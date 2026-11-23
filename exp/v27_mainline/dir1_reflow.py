"""方向 1 阶段 1B: 无锚点 CFG-Distilled Q32-Reflow + 单模型双档连续控制 γ ∈ [0,1].

核心设计:
  1) 离线构建 17,600 跨风格 + 2,200 Same = 19,800 条确定性教师轨迹 (6k Q32 旗舰):
     - z*_hiS: 6 步 e2full_d5 (强风格 + 93.3% Imp→Min)
     - z*_hiC: 6 步 csj_d6h   (高内容 DINO-C 0.636)
  2) 解析桥起点 (彻底拔掉 z_t4 拐杖):
     - z_0^perp = apply_latent_chroma_shift(z_s, tid, priors, alpha=1.0)
     - z_tilde_0 = (1 - 0.30) * eps + 0.30 * z_0^perp
  3) SLFUNetGamma: cond_mlp 末尾追加 1 维连续 γ (零初始化列, 无缝热启动 6k Q32 权重).
  4) Pseudo-Huber (c=0.1) + 潜空间 1 级 Haar 高频 L1 + Logit-Normal τ 采样,
     3500 updates 蒸馏进 4 步单分支 Euler.

用法 (4090):
  python -m exp.v27_mainline.dir1_reflow --teacher_anchor p1 --updates 3500
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift, build_or_load_v19_foundation
from exp.v19.routed_student import SpatiallyRoutedChordNet
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, match_scene_submode, sinusoidal_time_emb
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v27.icme_step2_closedset_accel import D5_CFG
from exp.v27.icme_step3_q32_vs_dino16 import make_bottleneck_net
from exp.v27_mainline import (
    B4_CKPT,
    OUT,
    P1_CKPT,
    Q32_FLAGSHIP_CKPT,
    ROOT,
    RawWriterV27M,
    build_wrong_task_index,
    run_measure_v27m,
)
from src.infer import canonicalizer, load_c3unet
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast as autocast, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc
from src.utils.structure import dwt2_haar, local_contrast_norm

ALPHA_BRIDGE = 0.55


class SLFUNetGamma(SLFUNet):
    """SLFUNet (Q32 32ch) extended with a zero-initialized 16D gamma embedding in [0, 1]."""

    def __init__(self, base_dim: int = 144, u_dim: int = 128, cond_dim: int = 288):
        super().__init__(base_dim=base_dim, u_dim=u_dim, cond_dim=cond_dim)
        self.dino_proj[0] = nn.Conv2d(32, 160, 1)
        in_dim = 64 + self.N_STYLE + u_dim + 2 + 16  # +16D sinusoidal gamma
        self.cond_mlp[0] = nn.Linear(in_dim, cond_dim)

    def load_from_q32_flagship(self, ckpt_path) -> None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["ema"]
        old_w = sd.pop("cond_mlp.0.weight")  # (288, 199)
        old_b = sd.pop("cond_mlp.0.bias")
        self.load_state_dict(sd, strict=False)
        with torch.no_grad():
            self.cond_mlp[0].weight[:, :199].copy_(old_w)
            self.cond_mlp[0].weight[:, 199:].zero_()
            self.cond_mlp[0].bias.copy_(old_b)

    def forward(
        self,
        z_tau: torch.Tensor,
        y16_lcn: torch.Tensor,
        q32_16: torch.Tensor,
        style_id: torch.Tensor,
        u_vivid: torch.Tensor,
        tau: torch.Tensor,
        *,
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

        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        d_feat = self.dino_proj(q32_16 * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, d_feat], dim=1)

        oh = F.one_hot(style_id.clamp(0, self.N_STYLE - 1), self.N_STYLE).float() * smask2
        uv = u_vivid * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        if gamma is None:
            g_val = torch.ones(B, device=dev, dtype=torch.float32)
        else:
            g_val = gamma.to(dev).float().view(B)
        g_emb = sinusoidal_time_emb(g_val, 16)
        cond = self.cond_mlp(torch.cat([temb, oh, uv, flags, g_emb], dim=-1))

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)

        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)

        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16, cond)

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        for blk in self.dec32:
            u32 = blk(u32, cond)

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond)

        return self.out_conv(F.silu(self.out_norm(u64)))


@torch.no_grad()
def _integrate_q32_seg(model, z_anc, eps, y16, d16_match, q32_in, tid, bank, *,
                       steps: int, full_steps: set[int], cfg_kw: dict,
                       wc_override: float | None = None, zero_vivid: bool = False) -> torch.Tensor:
    dev = eps.device
    B = len(eps)
    y_in, q_in, u_core, lscale = prepare_q32_cond_inputs(y16, d16_match, q32_in, tid, bank, lscale_abs=cfg_kw["lscale_abs"])
    if zero_vivid:
        u_core = torch.zeros_like(u_core)
    sc_src = F.normalize(F.avg_pool2d(d16_match.float(), 4).flatten(1), dim=-1)
    best_c = torch.einsum("bd,bkd->bk", sc_src, bank["sub_scene"].to(dev).float()[tid]).argmax(dim=1)
    ex_dsd = bank["sub_dsd"].to(dev).float()[tid, best_c]
    g_fig = torch.sigmoid((ex_dsd - float(bank["mid_dsd"])) * 550.0)
    gf4 = g_fig.view(B, 1, 1, 1)

    tau_v = (cfg_kw["tau_min"] + (cfg_kw["tau_hi"] - cfg_kw["tau_min"]) * g_fig).clamp(0.0, 0.78)
    if z_anc is None:
        if cfg_kw["tau_min"] != 0 or cfg_kw["tau_hi"] != 0:
            raise ValueError("An empty anchor requires tau_0=0")
        z = eps
    else:
        tau4 = tau_v.view(B, 1, 1, 1)
        z = (1.0 - tau4) * eps + tau4 * z_anc
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls0 = torch.zeros(B, device=dev)
    rescale_phi = float(cfg_kw.get("rescale_phi", 0.48))

    def _vel(zz, prog, use_full):
        tv = tau_v + prog * (1.0 - tau_v)
        gl = min(1.0, max(0.0, (prog - 0.36) / 0.32))
        ws_j = gf4 * (cfg_kw["ws_fig"] - 1.1 * gl) + (1.0 - gf4) * 1.4
        if use_full:
            with autocast(str(dev)):
                v_sc, v_s0, v_0c, v_00 = model(
                    zz.repeat(4, 1, 1, 1), y_in.repeat(4, 1, 1, 1), q_in.repeat(4, 1, 1, 1),
                    tid.repeat(4), u_core.repeat(4, 1), tv.repeat(4),
                    drop_style=torch.cat([f0, f0, f1, f1], 0),
                    drop_content=torch.cat([f0, f1, f0, f1], 0),
                    layout_scale=torch.cat([lscale, ls0, lscale, ls0], 0),
                ).float().chunk(4, 0)
            wc = (gf4 * (cfg_kw["wc_fig"] - 0.45 * gl) + (1.0 - gf4) * cfg_kw["wc_abs"]
                  if wc_override is None else wc_override)
            ws_p = gf4 * (gl * cfg_kw["wsp_fig_late"]) + (1.0 - gf4) * cfg_kw["wsp_abs"]
            vr = v_00 + wc * (v_0c - v_00) + ws_j * (v_sc - v_0c) + ws_p * (v_s0 - v_00)
            if rescale_phi > 0.0:
                s_ref = (gf4 * v_sc + (1.0 - gf4) * v_s0).flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                s_raw = vr.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                vr = rescale_phi * vr * (s_ref / s_raw) + (1.0 - rescale_phi) * vr
            return vr
        with autocast(str(dev)):
            v_sc, v_0c = model(
                zz.repeat(2, 1, 1, 1), y_in.repeat(2, 1, 1, 1), q_in.repeat(2, 1, 1, 1),
                tid.repeat(2), u_core.repeat(2, 1), tv.repeat(2),
                drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                layout_scale=lscale.repeat(2),
            ).float().chunk(2, 0)
        return v_0c + ws_j * (v_sc - v_0c)

    dt = (1.0 - tau_v).view(B, 1, 1, 1) / float(steps)
    for i in range(steps):
        uf = i in full_steps
        v1 = _vel(z, i / float(steps), uf)
        zm = z + 0.5 * dt * v1
        v2 = _vel(zm, (i + 0.5) / float(steps), uf)
        z = z + dt * v2
    return z


@torch.no_grad()
def prepare_q32_cond_inputs(
    y16: torch.Tensor,
    d16_match: torch.Tensor,
    q32_in: torch.Tensor,
    tid: torch.Tensor,
    bank: dict,
    lscale_abs: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact conditioning helper shared by 6k teacher, Stage 1B training/infer, and Dir 2 rollout."""
    dev = q32_in.device
    B = len(q32_in)
    _, u_core = match_scene_submode(d16_match, tid, bank)
    sc_src = F.normalize(F.avg_pool2d(d16_match.float(), 4).flatten(1), dim=-1)
    best_c = torch.einsum("bd,bkd->bk", sc_src, bank["sub_scene"].to(dev).float()[tid]).argmax(dim=1)
    ex_dsd = bank["sub_dsd"].to(dev).float()[tid, best_c]
    g_fig = torch.sigmoid((ex_dsd - float(bank["mid_dsd"])) * 550.0)
    lscale = (lscale_abs + (1.0 - lscale_abs) * g_fig).clamp(0.15, 1.0)
    c8_w = (1.0 - g_fig).clamp(0.0, 0.85).view(B, 1, 1, 1)
    f8 = F.avg_pool2d(q32_in, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    f8 = f8 - f8.mean(dim=(2, 3), keepdim=True)
    y8 = F.avg_pool2d(y16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    q_in = (1.0 - c8_w) * q32_in + c8_w * f8
    y_in = (1.0 - c8_w) * y16 + c8_w * y8
    return y_in, q_in, u_core, lscale


@torch.no_grad()
def _build_train_anchors(
    anchor_kind: str,
    bundle: dict,
    z_tr: torch.Tensor,
    d16_tr: torch.Tensor,
    c64_tr: torch.Tensor,
    src_idx_cross: torch.Tensor,
    tid_cross: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Compute teacher start anchors for 17,600 cross tasks on train_4400."""
    M = len(src_idx_cross)
    fnd = bundle["fnd"]
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}

    if anchor_kind == "chroma":
        out = torch.zeros((M, 4, 64, 64), dtype=torch.float16)
        for st in range(0, M, 64):
            sl = slice(st, min(st + 64, M))
            zs = z_tr[src_idx_cross[sl]].to(device).float()
            out[sl] = apply_latent_chroma_shift(zs, tid_cross[sl].to(device), priors, alpha=1.0).half().cpu()
        return out

    # B4 anchor (needed directly by b4 and as warm-start by p1)
    b4, _ = load_c3unet(B4_CKPT, device)
    canon = canonicalizer(device)
    out_b4 = torch.zeros((M, 4, 64, 64), dtype=torch.float16)
    for st in range(0, M, 64):
        sl = slice(st, min(st + 64, M))
        zs = z_tr[src_idx_cross[sl]].to(device).float()
        c_can = canon(zs, mode="c1_gentle", augment=False)
        out_b4[sl] = b4(c_can, 0.0, c_can, tid_cross[sl].to(device)).half().cpu()
    del b4, canon
    if anchor_kind == "b4":
        torch.cuda.empty_cache()
        return out_b4

    if anchor_kind == "p1":
        P_Y = priors["P_Y"]
        stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
        stu.load_state_dict(torch.load(P1_CKPT, map_location="cpu", weights_only=False)["model"])
        fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
        fiber_enc.load_state_dict(fnd["fiber_state"])
        u_all = fiber_enc.encode(fnd["tr_fib"].to(device).float())
        bary = fiber_enc.class_barycenters()
        del fiber_enc

        out_p1 = torch.zeros((M, 4, 64, 64), dtype=torch.float16)
        for st in range(0, M, 64):
            sl = slice(st, min(st + 64, M))
            si = src_idx_cross[sl]
            ti = tid_cross[sl].to(device)
            zs = z_tr[si].to(device).float()
            z0_b = out_b4[sl].to(device).float()
            z_anc = apply_latent_chroma_shift(z0_b, ti, priors, alpha=0.20)
            chord = bary[ti] - u_all[si]
            z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
            delta = stu(
                z_anc, z_luma, c64_tr[si].to(device).float(), d16_tr[si].to(device).float(),
                chord, same_mask=torch.zeros(len(si), dtype=torch.bool, device=device),
            )
            out_p1[sl] = (z_anc + delta).half().cpu()
        del stu
        torch.cuda.empty_cache()
        return out_p1

    raise ValueError(f"Unknown teacher_anchor={anchor_kind}")


@torch.no_grad()
def build_or_load_dir1_teacher_19800(
    bundle: dict,
    c21: dict,
    bank: dict,
    teacher_anchor: str = "p1",
    device: str = "cuda",
) -> dict:
    """Build or load 17,600 cross + 2,200 same = 19,800 Q32 flagship teacher trajectories."""
    cache_p = OUT / f"cache/dir1_teacher_19800_{teacher_anchor}.pt"
    if cache_p.exists():
        print(f"[Dir1 Cache] Loading {cache_p}", flush=True)
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    seed_all(20268001)
    tr = bundle["v18_cache"]["train"]
    z_tr = tr["z"]
    sid_tr = tr["sid"].long()
    d16_tr = tr["dino_16"]
    q32_tr = c21["train"]["q32"]
    c64_tr = tr["c_64"]
    y16_tr = coarse_y16_lcn(tr["y_64"].float()).half()
    N = len(z_tr)  # 4400

    # 17,600 cross + 2,200 same
    src_c, tid_c = [], []
    for delta in (1, 2, 3, 4):
        src_c.append(torch.arange(N, dtype=torch.long))
        tid_c.append((sid_tr + delta) % 5)
    src_cross = torch.cat(src_c, dim=0)
    tid_cross = torch.cat(tid_c, dim=0)
    M_cross = len(src_cross)

    src_same = torch.arange(0, N, 2, dtype=torch.long)  # 2,200
    tid_same = sid_tr[src_same]
    M_same = len(src_same)

    t0 = time.time()
    print(f"[Dir1 Cache] Building {teacher_anchor} anchors for {M_cross} cross tasks...", flush=True)
    anc_cross = _build_train_anchors(teacher_anchor, bundle, z_tr, d16_tr, c64_tr, src_cross, tid_cross, device)

    teacher = make_bottleneck_net(32).to(device).eval()
    teacher.load_state_dict(torch.load(Q32_FLAGSHIP_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)

    e2full_cfg = dict(D5_CFG, tau_min=0.30, tau_hi=0.68, ws_fig=5.5,
                      wc_fig=1.65, wc_abs=0.70, wsp_fig_late=3.6, wsp_abs=5.2)
    d6h_cfg = dict(D5_CFG, tau_min=0.48, tau_hi=0.78, ws_fig=5.2)

    gen = torch.Generator().manual_seed(20268002)
    eps_cross = torch.randn((M_cross, 4, 64, 64), generator=gen, dtype=torch.float32).half()
    eps_same = torch.randn((M_same, 4, 64, 64), generator=gen, dtype=torch.float32).half()

    z_hiS_c = torch.zeros((M_cross, 4, 64, 64), dtype=torch.float16)
    z_hiC_c = torch.zeros((M_cross, 4, 64, 64), dtype=torch.float16)

    for st in range(0, M_cross, 40):
        sl = slice(st, min(st + 40, M_cross))
        si = src_cross[sl]
        ti = tid_cross[sl].to(device)
        ab = anc_cross[sl].to(device).float()
        eb = eps_cross[sl].to(device).float()
        y16b = y16_tr[si].to(device).float()
        d16b = d16_tr[si].to(device).float()
        q32b = q32_tr[si].to(device).float()

        z_hiS_c[sl] = _integrate_q32_seg(
            teacher, ab, eb, y16b, d16b, q32b, ti, bank,
            steps=6, full_steps={0, 1}, cfg_kw=e2full_cfg,
        ).half().cpu()
        z_hiC_c[sl] = _integrate_q32_seg(
            teacher, ab, eb, y16b, d16b, q32b, ti, bank,
            steps=6, full_steps=set(), cfg_kw=d6h_cfg,
        ).half().cpu()
        if st == 0 or (st // 40) % 50 == 0 or (st + 40 >= M_cross):
            print(f"[Dir1 Cache] Teacher {min(st+40, M_cross):5d}/{M_cross} ({(time.time()-t0)/60:.1f} min)", flush=True)

    del teacher
    torch.cuda.empty_cache()

    z_same = z_tr[src_same].half()
    payload = {
        "teacher_anchor": teacher_anchor,
        "src_idx": torch.cat([src_cross, src_same], dim=0).to(torch.int32),
        "tid": torch.cat([tid_cross, tid_same], dim=0).to(torch.int8),
        "is_same": torch.cat([torch.zeros(M_cross, dtype=torch.bool), torch.ones(M_same, dtype=torch.bool)], dim=0),
        "z_hiS": torch.cat([z_hiS_c, z_same], dim=0),
        "z_hiC": torch.cat([z_hiC_c, z_same], dim=0),
        "eps": torch.cat([eps_cross, eps_same], dim=0),
    }
    torch.save(payload, cache_p)
    print(f"[Dir1 Cache] Saved {cache_p} ({len(payload['src_idx'])} pairs) in {(time.time()-t0)/60:.1f} min", flush=True)
    return payload


def train_dir1_reflow(
    bundle: dict,
    c21: dict,
    bank: dict,
    tcache: dict,
    updates: int = 3500,
    tag: str = "d1_reflow",
    device: str = "cuda",
) -> str:
    ckpt_p = OUT / f"{tag}.pt"
    if ckpt_p.exists():
        print(f"[Dir1 Train] Reusing {ckpt_p}", flush=True)
        return str(ckpt_p)

    seed_all(20268100)
    micro, accum = tier_micro_accum(96, default_micro_24gb=48, max_micro_12gb=16, device=device)

    tr = bundle["v18_cache"]["train"]
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float().view(1, 4, 1, 1)
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()
    z_tr = tr["z"]
    d16_tr = tr["dino_16"]
    q32_tr = c21["train"]["q32"]
    y16_tr = coarse_y16_lcn(tr["y_64"].float()).half()

    src_idx_all = tcache["src_idx"].long()
    tid_all = tcache["tid"].long()
    is_same_all = tcache["is_same"].bool()
    z_hiS_all = tcache["z_hiS"]
    z_hiC_all = tcache["z_hiC"]
    eps_all = tcache["eps"]
    M = len(src_idx_all)

    model = SLFUNetGamma(base_dim=144, u_dim=128).to(device)
    model.load_from_q32_flagship(Q32_FLAGSHIP_CKPT)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    t0 = time.time()
    print(f"[Dir1 Train] start {tag}: M={M}, micro={micro}×accum={accum}, updates={updates}", flush=True)

    for step in range(1, updates + 1):
        if step <= 150:
            lr = 1.0e-4 * (step / 150.0)
        else:
            prog = (step - 150) / max(1, updates - 150)
            lr = 1.0e-5 + 0.5 * (1.0e-4 - 1.0e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        ph_v, hf_v = 0.0, 0.0
        for _ in range(accum):
            idx = torch.randint(0, M, (micro,))
            si = src_idx_all[idx]
            ti = tid_all[idx].to(device, non_blocking=True)
            same_b = is_same_all[idx].to(device, non_blocking=True)
            B = len(idx)

            zs = z_tr[si].to(device, non_blocking=True).float()
            z_his = z_hiS_all[idx].to(device, non_blocking=True).float()
            z_hic = z_hiC_all[idx].to(device, non_blocking=True).float()

            # 80% paired Reflow noise + 20% fresh noise for path straightness & generalization
            ep_paired = eps_all[idx].to(device, non_blocking=True).float()
            use_fresh = (torch.rand(B, 1, 1, 1, device=device) < 0.20)
            ep = torch.where(use_fresh, torch.randn_like(ep_paired), ep_paired)

            # Analytical zero-param chroma bridge z_0^perp (alpha=1.0 on P_C, P_Y locked!)
            z0_perp = apply_latent_chroma_shift(zs, ti, priors, alpha=1.0)
            z0_perp = torch.where(same_b.view(B, 1, 1, 1), zs, z0_perp)
            z_tilde0 = (1.0 - ALPHA_BRIDGE) * ep + ALPHA_BRIDGE * z0_perp

            # Continuous gamma in [0, 1] with 50% exact endpoints {0, 1} + 50% uniform (0, 1)
            u_g = torch.rand(B, device=device)
            gamma = torch.where(u_g < 0.25, torch.zeros_like(u_g),
                    torch.where(u_g < 0.50, torch.ones_like(u_g), torch.rand(B, device=device)))
            g4 = gamma.view(B, 1, 1, 1)
            z_star = (1.0 - g4) * z_hic + g4 * z_his
            v_target = z_star - z_tilde0

            # Logit-normal tau (90%) + uniform [0, 0.98) (10%)
            u_t = torch.rand(B, device=device)
            tau_ln = torch.sigmoid(torch.randn(B, device=device) * 0.85)
            tau_unif = torch.rand(B, device=device) * 0.98
            tau = torch.where(u_t < 0.90, tau_ln, tau_unif).clamp(0.0, 0.995)
            t4 = tau.view(B, 1, 1, 1)
            z_tau = (1.0 - t4) * z_tilde0 + t4 * z_star

            y16b = y16_tr[si].to(device, non_blocking=True).float()
            d16b = d16_tr[si].to(device, non_blocking=True).float()
            q32b = q32_tr[si].to(device, non_blocking=True).float()
            y_in, q_in, u_core, lscale = prepare_q32_cond_inputs(y16b, d16b, q32b, ti, bank, lscale_abs=0.15)

            with autocast(device):
                v_pred = model(
                    z_tau, y_in, q_in, ti, u_core, tau,
                    layout_scale=lscale, gamma=gamma,
                )
                diff = v_pred.float() - v_target
                phuber = (torch.sqrt(diff.square().mean(dim=(1, 2, 3)) + 0.01) - 0.1).mean()
                z1_hat = (z_tau + (1.0 - t4) * v_pred.float()).clamp(-5.5, 5.5)
                _, lh_p, hl_p, hh_p = dwt2_haar(z1_hat)
                _, lh_t, hl_t, hh_t = dwt2_haar(z_star)
                hf_loss = ((lh_p - lh_t).abs() + (hl_p - hl_t).abs() + (hh_p - hh_t).abs()).mean() / 3.0
                yh_32 = F.avg_pool2d(((z1_hat * w_luma).sum(1, keepdim=True) + b_luma).clamp(0.0, 1.0), 2)
                yt_32 = F.avg_pool2d(((z_star * w_luma).sum(1, keepdim=True) + b_luma).clamp(0.0, 1.0), 2)
                st_loss = F.smooth_l1_loss(local_contrast_norm(yh_32, 5, 0.03), local_contrast_norm(yt_32, 5, 0.03), beta=0.05)
                kin_loss = v_pred.float().square().mean()
                loss = (phuber + 0.35 * hf_loss + 0.25 * st_loss + 1e-3 * kin_loss) / accum

            loss.backward()
            ph_v += float(phuber.item()) / accum
            hf_v += float(hf_loss.item()) / accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.997).add_(pw, alpha=0.003)

        if step == 1 or step % 350 == 0 or step == updates:
            print(f"[Dir1 {tag}] {step:4d}/{updates} phuber={ph_v:.4f} hf={hf_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({
        "ema": ema.state_dict(),
        "updates": updates,
        "alpha_bridge": ALPHA_BRIDGE,
        "teacher_anchor": tcache["teacher_anchor"],
    }, ckpt_p)
    print(f"[Dir1 Train] Saved {ckpt_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return str(ckpt_p)


@torch.no_grad()
def integrate_dir1_reflow(
    model: SLFUNetGamma,
    z_src: torch.Tensor,
    eps: torch.Tensor,
    y16: torch.Tensor,
    d16_match: torch.Tensor,
    q32_in: torch.Tensor,
    tid: torch.Tensor,
    priors: dict,
    bank: dict,
    *,
    steps: int = 4,
    solver: str = "euler",
    gamma_val: float = 1.0,
    same_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    dev = z_src.device
    B = len(z_src)
    y_in, q_in, u_core, lscale = prepare_q32_cond_inputs(y16, d16_match, q32_in, tid, bank, lscale_abs=0.15)
    z0_perp = apply_latent_chroma_shift(z_src.float(), tid, priors, alpha=1.0)
    if same_mask is not None:
        z0_perp = torch.where(same_mask.view(B, 1, 1, 1), z_src.float(), z0_perp)
    z = (1.0 - ALPHA_BRIDGE) * eps.float() + ALPHA_BRIDGE * z0_perp
    gamma = torch.full((B,), float(gamma_val), device=dev)
    dt = 1.0 / float(steps)

    for i in range(steps):
        tau = torch.full((B,), i * dt, device=dev)
        with autocast(str(dev)):
            v1 = model(z, y_in, q_in, tid, u_core, tau, layout_scale=lscale, gamma=gamma).float()
        if solver == "rk2":
            zm = z + 0.5 * dt * v1
            tau_m = torch.full((B,), (i + 0.5) * dt, device=dev)
            with autocast(str(dev)):
                v2 = model(zm, y_in, q_in, tid, u_core, tau_m, layout_scale=lscale, gamma=gamma).float()
            z = z + dt * v2
        else:
            z = z + dt * v1
    return z


@torch.no_grad()
def eval_dir1_reflow(ckpt_p: str, bundle: dict, c21: dict, bank: dict, prefix: str = "D1v2", device: str = "cuda") -> dict:
    tasks, z_src_750, _ = tasks_and_sources()
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    q32_750 = c21["test750"]["q32"]
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    dec = load_s10_decoder(device)

    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    tid_wrong_all, _ = build_wrong_task_index(tasks, sid_all, tid_all)

    model = SLFUNetGamma(base_dim=144, u_dim=128).to(device).eval()
    model.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=False)["ema"], strict=True)

    eval_specs = [
        (f"{prefix}_g10_e4",    4, "euler", 1.0, False),
        (f"{prefix}_g05_e4",    4, "euler", 0.5, False),
        (f"{prefix}_g00_e4",    4, "euler", 0.0, False),
        (f"{prefix}_wrong_e4",  4, "euler", 1.0, True),
        (f"{prefix}_g10_e2",    2, "euler", 1.0, False),
        (f"{prefix}_g10_rk2_4", 4, "rk2",   1.0, False),
    ]
    out = {"task": "Dir 1 Stage 1B: Anchor-Free 4-Step Q32-Reflow", "ckpt": str(ckpt_p), "updated_utc": utc(), "arms": {}}

    for arm_name, steps, solver, gval, is_wrong in eval_specs:
        m_dir = OUT / "measured" / arm_name
        if not (m_dir / "summary_v27m.json").exists():
            wr = RawWriterV27M(arm_name, dec=dec, device=device)
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                tid_u = (tid_wrong_all[sl] if is_wrong else tid_all[sl]).to(device)
                sm = (torch.zeros_like(same_all[sl]) if is_wrong else same_all[sl]).to(device)
                zg = integrate_dir1_reflow(
                    model, z_src_750[sl].to(device).float(), eps_750[sl].to(device).float(),
                    y16_750[sl].to(device).float(), d16_750[sl].to(device).float(),
                    q32_750[sl].to(device).float(), tid_u, priors, bank,
                    steps=steps, solver=solver, gamma_val=gval, same_mask=sm,
                )
                wr.write_batch(
                    torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), zg),
                    tasks[sl],
                )
            wr.finalize(arm=arm_name, steps=steps, solver=solver, gamma=gval, is_wrong=is_wrong)
        out["arms"][arm_name] = run_measure_v27m(arm_name, device=device)
        save_json(OUT / "dir1_reflow_eval.json", out)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--teacher_anchor", default="p1", choices=["p1", "b4", "chroma"])
    ap.add_argument("--updates", type=int, default=3500)
    ap.add_argument("--tag", default="d1_reflow_v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)

    pack = build_or_load_v21_cache(args.pool, device=args.device)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    tcache = build_or_load_dir1_teacher_19800(bundle, c21, bank, teacher_anchor=args.teacher_anchor, device=args.device)
    ckpt_p = train_dir1_reflow(bundle, c21, bank, tcache, updates=args.updates, tag=args.tag, device=args.device)
    eval_dir1_reflow(ckpt_p, bundle, c21, bank, prefix="D1v2", device=args.device)
    print("[Dir1] ALL DONE →", OUT / "dir1_reflow_eval.json", flush=True)


if __name__ == "__main__":
    main()
