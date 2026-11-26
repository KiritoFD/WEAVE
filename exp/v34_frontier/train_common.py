"""Shared V34 training, continuous style-instance cache, diversity probe, and evaluation helpers.

Maintains 100% Zero-DINO AST hygiene during training and inference.
"""
from __future__ import annotations

import copy
import gc
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v27_single.bare_q32 import load_cache
from exp.v28_unified.student_models import build_student_model
from exp.v29_biglittle import POOL_C_PATH, assert_vram_budget, atomic_save
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v29_biglittle.phase1_train import (
    _HighTEndpointSWDCritic,
    _blur_or_drop_y16,
    _build_style_counts,
    _pseudo_huber_velocity,
)
from exp.v29_biglittle.phase2_tune import default_knob_schedule
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.assets import build_or_load_q64_hi32_cache, build_or_load_s2_pert_cache
from exp.v33_semantic_control.models import (
    PCoreWithSemanticControl,
    SemanticControlAdapter,
    integrate_v33_trajectory,
)
from exp.v34_frontier import (
    OUT,
    SEED_V34,
    V32_ECORE_CKPT,
    RawWriterV34,
    measure_and_summarize_v34,
)
from exp.v34_frontier.stage_a_startpoint import build_or_load_analytic_priors, reconstruct_analytic_draft
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import auto_vram_budget_gb, bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc, vram_total_gb
from src.utils.structure import dwt2_haar, rgb_to_gray


class MultiScaleTextureExtractor(nn.Module):
    """Frozen multi-scale ConvNeXt-Tiny channel std (96+192+384) + 2-level Haar wavelet energy (12) = 684-D."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        cnx = load_convnext_tiny_offline(device)
        self.s1 = cnx[:2].eval()
        self.s2 = cnx[2:4].eval()
        self.s3 = cnx[4:6].eval()
        for mod in (self.s1, self.s2, self.s3):
            for p in mod.parameters():
                p.requires_grad_(False)
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(rgb01.float().clamp(0.0, 1.0), size=(256, 256), mode="bilinear", align_corners=False)
        xn = (x - self.im_mean.to(x.device)) / self.im_std.to(x.device)
        f1 = self.s1(xn).float()
        f2 = self.s2(f1).float()
        f3 = self.s3(f2).float()
        std1 = f1.std(dim=(2, 3), unbiased=False)  # (B, 96)
        std2 = f2.std(dim=(2, 3), unbiased=False)  # (B, 192)
        std3 = f3.std(dim=(2, 3), unbiased=False)  # (B, 384)

        g1 = rgb_to_gray(x)
        ll1, lh1, hl1, hh1 = dwt2_haar(g1)
        ll2, lh2, hl2, hh2 = dwt2_haar(ll1)

        def _band_feats(lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> list[torch.Tensor]:
            eh = torch.log1p(lh.abs().mean(dim=(1, 2, 3)) * 10.0)
            ev = torch.log1p(hl.abs().mean(dim=(1, 2, 3)) * 10.0)
            ed = torch.log1p(hh.abs().mean(dim=(1, 2, 3)) * 10.0)
            tot = eh + ev + ed + 1e-6
            hv_ratio = (eh - ev) / tot
            hf_total = torch.log1p((lh.square().mean(dim=(1, 2, 3)) + hl.square().mean(dim=(1, 2, 3)) + hh.square().mean(dim=(1, 2, 3))).sqrt() * 10.0)
            diag_ratio = ed / tot
            return [eh, ev, ed, hv_ratio, hf_total, diag_ratio]

        haar_feats = torch.stack(_band_feats(lh1, hl1, hh1) + _band_feats(lh2, hl2, hh2), dim=-1)  # (B, 12)
        return torch.cat([std1, std2, std3, haar_feats], dim=-1)  # (B, 684)


@torch.no_grad()
def build_or_load_style_instances(device: str = "cuda") -> dict[str, Any]:
    """Build or load train_4400-only instance-level style statistics & 684-D multi-scale texture cache."""
    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "v34_style_instances.pt"
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)

    sanitize_cuda_env()
    seed_all(SEED_V34 + 10)
    t0 = time.monotonic()

    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr = v18["train"]
    sid_all = tr["sid"].long().cpu()
    inst_417 = F.normalize(cnx_bank["norm_417_all"].float().cpu(), dim=-1)  # (4400, 417)
    inst_scene = F.normalize(F.avg_pool2d(stiefel["train"]["cnx_gray"].float().cpu(), 4).flatten(1), dim=-1)  # (4400, 6144)

    dec = load_s10_decoder(device).to(torch.bfloat16).eval().requires_grad_(False)
    tex_ext = MultiScaleTextureExtractor(device=device).eval()
    tex_chunks = []
    for st in range(0, len(tr["z"]), 32):
        sl = slice(st, min(st + 32, len(tr["z"])))
        rgb = decode_s10(dec, tr["z"][sl].to(device).float(), dtype=torch.bfloat16, batch=sl.stop - sl.start, output_cpu=False).clamp(0.0, 1.0)
        tex_chunks.append(tex_ext(rgb).cpu())
    tex_raw = torch.cat(tex_chunks, 0)  # (4400, 684)
    del dec, tex_ext
    gc.collect()
    torch.cuda.empty_cache()

    tex_mu = tex_raw.mean(dim=0, keepdim=True)
    tex_sd = tex_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    inst_tex = F.normalize((tex_raw - tex_mu) / tex_sd, dim=-1)  # (4400, 684)
    inst_1101 = torch.cat([inst_417, inst_tex], dim=-1)  # 1:1 balanced energy between 417-D and 684-D

    # Per-style indices and top-8 same-style neighbor indices in scene+style space (for continuous training mixup)
    style_indices: list[torch.Tensor] = []
    knn8 = torch.zeros(len(sid_all), 8, dtype=torch.long)
    knn_sim = inst_417 @ inst_417.T
    for s in range(5):
        s_idx = (sid_all == s).nonzero(as_tuple=True)[0]
        style_indices.append(s_idx)
        sub_sim = knn_sim[s_idx][:, s_idx]
        sub_sim.fill_diagonal_(-1e9)
        top8 = sub_sim.topk(8, dim=1).indices
        knn8[s_idx] = s_idx[top8]

    # Pick two maximally separated representative exemplars per style (excluding outliers via scene-density medoid check)
    extreme_pairs = torch.zeros(5, 2, dtype=torch.long)
    for s in range(5):
        s_idx = style_indices[s]
        feats = F.normalize(inst_1101[s_idx], dim=-1)
        sim_mat = feats @ feats.T
        # Keep core 70% highest mean cosine to avoid freak corruptions, then pick farthest pair
        mean_sim = sim_mat.mean(dim=1)
        keep_mask = mean_sim >= torch.quantile(mean_sim, 0.30)
        core_sub = s_idx[keep_mask]
        core_feats = feats[keep_mask]
        pair_sim = core_feats @ core_feats.T
        flat_idx = int(pair_sim.argmin().item())
        i_a, i_b = divmod(flat_idx, len(core_sub))
        extreme_pairs[s, 0] = core_sub[i_a]
        extreme_pairs[s, 1] = core_sub[i_b]

    payload = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "sid": sid_all,
        "inst_417": inst_417.half(),
        "inst_tex": inst_tex.half(),
        "inst_1101": inst_1101.half(),
        "inst_scene": inst_scene.half(),
        "tex_mu": tex_mu,
        "tex_sd": tex_sd,
        "style_indices": style_indices,
        "knn8": knn8,
        "extreme_pairs": extreme_pairs,
    }
    torch.save(payload, p)
    print(f"[V34 InstCache] Built train_4400 continuous instance cache in {payload['wall_sec']:.1f}s -> {p}", flush=True)
    return payload


def retrieve_continuous_style_u(
    cnx_gray_src: torch.Tensor,
    tid: torch.Tensor,
    inst_cache: dict[str, Any],
    *,
    cond_key: str = "inst_417",
    top_m: int = 6,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Continuous convex combination over top-m scene-matched training instances in target style tid."""
    dev = cnx_gray_src.device
    sc_src = F.normalize(F.avg_pool2d(cnx_gray_src.float(), 4).flatten(1), dim=-1)
    u_bank = inst_cache[cond_key].to(dev).float()
    scene_bank = inst_cache["inst_scene"].to(dev).float()
    style_indices = inst_cache["style_indices"]

    out = torch.empty(len(tid), u_bank.shape[1], device=dev, dtype=torch.float32)
    for s in range(5):
        mask = tid == s
        if not mask.any():
            continue
        s_idx = style_indices[s].to(dev)
        sims = sc_src[mask] @ scene_bank[s_idx].T  # (Bs, N_s)
        k = min(top_m, sims.shape[1])
        top_vals, top_pos = sims.topk(k, dim=1)
        weights = F.softmax(top_vals / max(temperature, 1e-4), dim=1)  # (Bs, k)
        picked_idx = s_idx[top_pos]  # (Bs, k)
        out[mask] = (weights.unsqueeze(-1) * u_bank[picked_idx]).sum(dim=1)
    return out


def schedule_for_v34(knob_s: float) -> dict[str, Any]:
    """V32 knob schedule augmented with matching CFG-rebalanced tau0 & solver kwargs."""
    kw = {k: v for k, v in default_knob_schedule(knob_s, use_gamma=True).items() if k != "draft_lambda"}
    # Keep V32 calibrated tau0 curve
    if abs(knob_s - 0.65) < 0.03:
        kw["tau0"] = 0.87
        kw["draft_lambda"] = 0.55
    elif abs(knob_s - 0.75) < 0.03:
        kw["tau0"] = 0.835
        kw["draft_lambda"] = 0.65
    elif abs(knob_s - 0.85) < 0.03:
        kw["tau0"] = 0.79
        kw["draft_lambda"] = 0.75
    elif abs(knob_s - 1.00) < 0.03:
        kw["tau0"] = 0.635
        kw["draft_lambda"] = 0.90
    elif abs(knob_s - 1.25) < 0.03:
        kw["tau0"] = 0.50
        kw["draft_lambda"] = 1.00
    elif abs(knob_s - 1.50) < 0.03:
        kw["tau0"] = 0.36
        kw["draft_lambda"] = 1.00
    else:
        kw["draft_lambda"] = 0.75
    return kw


def load_pcore_from_v34_ckpt(ckpt_path: Path, device: str = "cuda") -> tuple[nn.Module, dict[str, Any]]:
    """Reconstruct 22M/44M P-Core (with optional SemanticControlAdapter) from V34 checkpoint."""
    sv = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sv["cfg"]
    pcore = PCoreSLFUNet(
        base_dim=int(cfg["base_dim"]),
        u_dim=int(cfg["u_dim"]),
        cond_dim=int(cfg["cond_dim"]),
        use_gamma=True,
        normalize_u=True,
    ).to(device)
    if cfg.get("use_adapter", False):
        w0 = int(cfg["base_dim"])
        ctrl = SemanticControlAdapter(
            d_q16=int(cfg.get("d_q16", 32)),
            d_q32=int(cfg.get("d_q32", 32)),
            w32=int(cfg.get("ctrl_w32", 144)),
            w16=int(cfg.get("ctrl_w16", 256)),
            pcore_w32=w0 * 2,
            pcore_w16=w0 * 3,
            cond_dim=int(cfg["cond_dim"]),
        ).to(device)
        model: nn.Module = PCoreWithSemanticControl(pcore, ctrl).to(device)
    else:
        model = pcore
    model.load_state_dict(sv["ema"], strict=True)
    model.eval().requires_grad_(False)
    return model, cfg


def build_pcore_v34(
    *,
    base_dim: int,
    u_dim: int,
    cond_dim: int,
    use_adapter: bool,
    d_q16: int = 32,
    d_q32: int = 32,
    ctrl_w32: int = 144,
    ctrl_w16: int = 256,
    device: str = "cuda",
) -> tuple[nn.Module, dict[str, Any]]:
    """Instantiate 22M/44M P-Core with optional 7M SemanticControlAdapter."""
    pcore = PCoreSLFUNet(base_dim=base_dim, u_dim=u_dim, cond_dim=cond_dim, use_gamma=True, normalize_u=True).to(device)
    pcore_params = pcore.n_params()
    if use_adapter:
        ctrl = SemanticControlAdapter(
            d_q16=d_q16,
            d_q32=d_q32,
            w32=ctrl_w32,
            w16=ctrl_w16,
            pcore_w32=base_dim * 2,
            pcore_w16=base_dim * 3,
            cond_dim=cond_dim,
        ).to(device)
        ctrl_params = ctrl.n_params()
        model: nn.Module = PCoreWithSemanticControl(pcore, ctrl).to(device)
    else:
        ctrl_params = 0
        model = pcore
    total_params = sum(p.numel() for p in model.parameters())
    cfg = {
        "base_dim": base_dim,
        "u_dim": u_dim,
        "cond_dim": cond_dim,
        "use_adapter": use_adapter,
        "d_q16": d_q16,
        "d_q32": d_q32,
        "ctrl_w32": ctrl_w32,
        "ctrl_w16": ctrl_w16,
        "pcore_params": pcore_params,
        "ctrl_params": ctrl_params,
        "total_params": total_params,
    }
    return model, cfg


def run_pcore_train_step(
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    ema: nn.Module,
    params_list: list[nn.Parameter],
    swd_critic: _HighTEndpointSWDCritic,
    w_luma: torch.Tensor,
    b_luma: torch.Tensor,
    z1: torch.Tensor,
    zt: torch.Tensor,
    target_v: torch.Tensor,
    y16: torch.Tensor,
    y16_aug: torch.Tensor,
    y16_clean_m: torch.Tensor,
    q16_aug: torch.Tensor,
    tid: torch.Tensor,
    u_in: torch.Tensor,
    tau: torch.Tensor,
    ds: torch.Tensor,
    dc: torch.Tensor,
    lscale: torch.Tensor,
    gamma_b: torch.Tensor,
    micro: int,
    accum: int,
    swd_micro: int,
    step: int,
    seed_base: int,
    use_adapter: bool = False,
    q32_hi_aug: torch.Tensor | None = None,
    device: str = "cuda",
) -> tuple[float, float, float, float]:
    """Execute one micro-batched PHuber + Y16 + norm-balanced SWD critic training step and EMA update."""
    opt.zero_grad(set_to_none=True)
    phuber_acc, st_acc = 0.0, 0.0
    for m in range(accum):
        sl = slice(m * micro, (m + 1) * micro)
        q32_sl = q32_hi_aug[sl] if q32_hi_aug is not None else None
        with bf16_autocast(device):
            pred_v = _forward_model(
                model, zt[sl], y16_aug[sl], q16_aug[sl], tid[sl], u_in[sl], tau[sl],
                use_adapter=use_adapter, q32_hi=q32_sl,
                drop_style=ds[sl], drop_content=dc[sl], layout_scale=lscale[sl], gamma=gamma_b[sl],
            )
            phuber_m = _pseudo_huber_velocity(pred_v, target_v[sl], delta=0.05)
            z1_hat = (zt[sl] + (1.0 - tau[sl]).view(-1, 1, 1, 1) * pred_v.float()).clamp(-5.5, 5.5)
            y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
            w_st = (tau[sl] > 0.25).float() * (~dc[sl]).float() * y16_clean_m[sl].float() * (gamma_b[sl] > 0.75).float()
            st_m = ((y16_hat - y16[sl]).abs().mean(dim=(1, 2, 3)) * w_st).sum() / w_st.sum().clamp_min(1.0)
            loss_m = (phuber_m + 0.10 * st_m) / float(accum)
        loss_m.backward()
        phuber_acc += float(phuber_m.item()) / float(accum)
        st_acc += float(st_m.item()) / float(accum)

    pix_val = 0.0
    if step % 2 == 0:
        hi_idx = ((tau >= 0.60) & (tau <= 0.95) & (~dc)).nonzero().flatten()
        if len(hi_idx) >= 2:
            pick = hi_idx[:6]
            n_pick = len(pick)
            g_main = [p.grad.detach().clone() if p.grad is not None else None for p in params_list]
            norm_main = math.sqrt(sum(float(g.square().sum().item()) for g in g_main if g is not None))
            opt.zero_grad(set_to_none=True)

            pg_s = torch.Generator(device=device).manual_seed(seed_base + 9000 + step)
            proj = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=pg_s), mode="reduced").Q for c in (64, 128, 256))
            for s_st in range(0, n_pick, swd_micro):
                sub = pick[s_st:s_st + swd_micro]
                w_sub = float(len(sub)) / float(n_pick)
                with bf16_autocast(device):
                    q32_pick = q32_hi_aug[sub] if q32_hi_aug is not None else None
                    pv_hi = _forward_model(
                        model, zt[sub], y16_aug[sub], q16_aug[sub], tid[sub], u_in[sub], tau[sub],
                        use_adapter=use_adapter, q32_hi=q32_pick,
                        drop_style=ds[sub], drop_content=dc[sub], layout_scale=lscale[sub], gamma=gamma_b[sub],
                    )
                    z1_hi = (zt[sub] + (1.0 - tau[sub]).view(-1, 1, 1, 1) * pv_hi.float()).clamp(-5.5, 5.5)
                    l_pix_sub = swd_critic.loss(z1_hi, z1[sub], proj) * w_sub
                    pix_val += float(l_pix_sub.item())
                l_pix_sub.backward()

            norm_pix = math.sqrt(sum(float(p.grad.square().sum().item()) for p in params_list if p.grad is not None))
            scale = min(2.0, (0.25 * norm_main) / max(norm_pix, 1e-6))
            for p, gm in zip(params_list, g_main):
                if p.grad is not None and gm is not None:
                    p.grad.mul_(scale).add_(gm)
                elif gm is not None:
                    p.grad = gm
            del g_main

    gn = torch.nn.utils.clip_grad_norm_(params_list, 1.0, error_if_nonfinite=True)
    opt.step()
    with torch.no_grad():
        torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in params_list], 0.004)
    return phuber_acc, st_acc, pix_val, float(gn)


def _forward_model(
    model: nn.Module,
    zt: torch.Tensor,
    y16: torch.Tensor,
    q16: torch.Tensor,
    tid: torch.Tensor,
    u_in: torch.Tensor,
    tau: torch.Tensor,
    *,
    use_adapter: bool,
    q32_hi: torch.Tensor | None = None,
    drop_style: torch.Tensor,
    drop_content: torch.Tensor,
    layout_scale: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    if use_adapter:
        return model(
            zt, y16, q16, tid, u_in, tau,
            q32_hi=q32_hi,
            drop_style=drop_style,
            drop_content=drop_content,
            layout_scale=layout_scale,
            gamma=gamma,
        )
    return model(
        zt, y16, q16, tid, u_in, tau,
        drop_style=drop_style,
        drop_content=drop_content,
        layout_scale=layout_scale,
        gamma=gamma,
    )


def sample_analytic_bridge_segment(
    z1: torch.Tensor,
    z0: torch.Tensor,
    ze_analytic: torch.Tensor,
    tau0: torch.Tensor,
    s_prog: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact ODE bridge (tau, z_tau, target_v) from analytic draft start at tau0 to z1 at 1.0."""
    t0_4 = tau0.view(-1, 1, 1, 1)
    s_4 = s_prog.view(-1, 1, 1, 1)
    z_start = (1.0 - t0_4) * z0 + t0_4 * ze_analytic
    tau = tau0 + s_prog * (1.0 - tau0)
    zt = (1.0 - s_4) * z_start + s_4 * z1
    denom = (1.0 - t0_4).clamp_min(0.15)
    target_v = (z1 - z_start) / denom
    return tau, zt, target_v


def train_pcore_v34(
    arm_name: str,
    run_dir: Path,
    *,
    base_dim: int = 144,
    cond_dim: int = 320,
    style_mode: str = "k8",      # "k8" | "inst_417" | "inst_1101"
    use_adapter: bool = False,   # True => SemanticControlAdapter (V33 S2)
    analytic_mode: str = "A0",   # "A0" | "A1" | "A2" | "A3" (optional target blend on E-Core / analytic start)
    updates: int = 3500,
    lr: float = 3.0e-4,
    batch_size: int = 128,
    seed_offset: int = 0,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train a V34 arm with shared data, updates, and loss objective, with 500-step atomic resume."""
    sanitize_cuda_env()
    seed_all(SEED_V34 + seed_offset)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_p = run_dir / f"{arm_name}_final.pt"
    resume_p = run_dir / f"{arm_name}_resume.pt"
    if ckpt_p.exists() and smoke_steps == 0:
        sv = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        print(f"[{arm_name}] Reusing completed checkpoint {ckpt_p}", flush=True)
        return {
            "arm": arm_name,
            "ckpt": str(ckpt_p),
            "cfg": sv["cfg"],
            "train_min": sv.get("train_min", 0.0),
            "peak_vram_gb": sv.get("peak_vram_gb", 0.0),
        }

    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    q16_all = stiefel["train"]["q32"]
    norm417_all = cnx_bank["norm_417_all"]
    sub_raw417 = cnx_bank["sub_raw417"]
    core_cls = [ci.cpu().long() for ci in cnx_bank["core_idx_per_style"]]

    inst_cache = build_or_load_style_instances(device=device) if style_mode != "k8" else None
    if inst_cache is not None:
        u_inst_all = inst_cache[style_mode]  # (4400, 417) or (4400, 1101)
        knn8_all = inst_cache["knn8"]
        u_dim = int(u_inst_all.shape[1])
    else:
        u_inst_all, knn8_all, u_dim = None, None, 417

    q64_cache = build_or_load_q64_hi32_cache(device=device) if use_adapter else None
    q32_hi_all = q64_cache["train"]["q32_hi"] if q64_cache is not None else None
    pert_vars = build_or_load_s2_pert_cache(n_variants=2, device=device)["variants"] if use_adapter else None

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr = v18["train"]
    w_luma = v18["luma_proj"]["w"].to(device).float()
    b_luma = v18["luma_proj"]["b"].to(device).float()
    z_all = tr["z"]
    sid_all = tr["sid"].long()
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    all_cls = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]

    pool_c = torch.load(POOL_C_PATH, map_location="cpu", weights_only=False)
    pc_y16 = coarse_y16_lcn(pool_c["y_64"].float()).half()
    pc_zreal = pool_c["z_real"]
    pc_tid = pool_c["tid"].long()
    pc_ridx = pool_c["real_idx"].long()
    pc_q16 = q16_all[pc_ridx]
    pc_q32_hi = q32_hi_all[pc_ridx] if q32_hi_all is not None else None

    swd_critic = _HighTEndpointSWDCritic(device=device)
    swd_critic.set_moment_scale(torch.cat([z_all[ci[:10]] for ci in core_cls], dim=0))

    ecore_tr, raw_stats_tr, priors_ec_tr, priors_a_tr, c64_tr = None, None, None, None, None
    if analytic_mode != "A0":
        ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
        raw_stats_tr = ec_sv["raw_stats"].to(device).float()
        priors_ec_tr = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
        ecore_tr = build_student_model(ec_sv.get("arch", "unet"), raw_stats_tr.shape[-1], device=device).eval().requires_grad_(False)
        ecore_tr.load_state_dict(ec_sv["ema"], strict=True)
        priors_a_tr = build_or_load_analytic_priors(priors_ec_tr)
        c64_tr = tr["c_64"]

    model, cfg = build_pcore_v34(
        base_dim=base_dim,
        u_dim=u_dim,
        cond_dim=cond_dim,
        use_adapter=use_adapter,
        device=device,
    )
    cfg.update({"style_mode": style_mode, "analytic_mode": analytic_mode, "updates": updates})
    model.train().requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1.0e-4)
    params_list = list(model.parameters())

    start_step = 0
    if resume_p.exists() and smoke_steps == 0:
        r_sv = torch.load(resume_p, map_location="cpu", weights_only=False)
        model.load_state_dict(r_sv["model"], strict=True)
        ema.load_state_dict(r_sv["ema"], strict=True)
        opt.load_state_dict(r_sv["opt"])
        start_step = int(r_sv["step"])
        print(f"[{arm_name}] Resuming from step {start_step}/{updates}", flush=True)

    # Saturate 24GB 4090 cleanly (128x1 for 22M, 64x2 for 44M/68M) while keeping 12GB/8GB safe
    default_micro_24 = 64 if base_dim >= 192 else 128
    micro, accum = tier_micro_accum(batch_size, default_micro_24gb=default_micro_24, max_micro_12gb=32, max_micro_8gb=16, device=device)
    swd_micro = (6 if base_dim <= 224 else 3) if vram_total_gb(device) > 13.5 else 2
    vram_budget_gb = auto_vram_budget_gb(device)
    torch.cuda.empty_cache()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    n_pc = int(round(batch_size * 0.15))
    n_self = batch_size - n_pc
    self_counts = _build_style_counts(n_self)
    target_steps = smoke_steps if smoke_steps > 0 else updates
    t0 = time.monotonic()

    print(
        f"[{arm_name}] Start train | params={cfg['total_params']/1e6:.2f}M (pcore={cfg['pcore_params']/1e6:.2f}M, ctrl={cfg['ctrl_params']/1e6:.2f}M) "
        f"| style={style_mode} | adapter={use_adapter} | analytic={analytic_mode} | micro={micro}x{accum}",
        flush=True,
    )

    for step in range(start_step + 1, target_steps + 1):
        if step <= 200:
            cur_lr = lr * (step / 200.0)
        else:
            frac = (step - 200) / float(max(1, updates - 200))
            cur_lr = 2.0e-5 + 0.5 * (lr - 2.0e-5) * (1.0 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        pools = core_cls if (step % 5 != 0) else all_cls
        idx_self = torch.cat([ci[torch.randint(0, len(ci), (cnt,))] for ci, cnt in zip(pools, self_counts)], dim=0)
        p_pick = torch.randint(0, len(pc_tid), (n_pc,))

        z1 = torch.cat([z_all[idx_self], pc_zreal[p_pick]], dim=0).to(device, non_blocking=True).float()
        tid = torch.cat([sid_all[idx_self], pc_tid[p_pick]], dim=0).to(device, non_blocking=True)
        q16 = torch.cat([q16_all[idx_self], pc_q16[p_pick]], dim=0).to(device, non_blocking=True).float()
        y16 = torch.cat([y16_all[idx_self], pc_y16[p_pick]], dim=0).to(device, non_blocking=True).float()

        real_indices = torch.cat([idx_self, pc_ridx[p_pick]], dim=0)
        if style_mode == "k8":
            u_ex = norm417_all[real_indices].to(device, non_blocking=True).float()
            sub_k = torch.randint(0, 8, (batch_size,))
            u_sub = sub_raw417[tid.cpu(), sub_k].to(device, non_blocking=True).float()
            u_in = torch.where(torch.rand(batch_size, 1, device=device) < 0.75, u_ex, u_sub)
        else:
            assert u_inst_all is not None and knn8_all is not None
            u_self = u_inst_all[real_indices].to(device, non_blocking=True).float()
            nbr_pick = torch.randint(0, 8, (batch_size,))
            u_nbr = u_inst_all[knn8_all[real_indices, nbr_pick]].to(device, non_blocking=True).float()
            w_mix = torch.distributions.Beta(3.0, 1.0).sample((batch_size, 1)).to(device).clamp(0.50, 1.0)
            u_blend = w_mix * u_self + (1.0 - w_mix) * u_nbr
            use_pure = (torch.rand(batch_size, 1, device=device) < 0.50)
            u_in = torch.where(use_pure, u_self, u_blend)

        if use_adapter:
            assert q32_hi_all is not None and pert_vars is not None
            q32_hi = torch.cat([q32_hi_all[idx_self], pc_q32_hi[p_pick]], dim=0).to(device, non_blocking=True).float()
            p_mask = torch.rand(batch_size) < 0.30
            if bool(p_mask.any()):
                p_pos = p_mask.nonzero().flatten()
                r_ids = real_indices[p_pos]
                v_pick = pert_vars[int(step % len(pert_vars))]
                q16[p_pos.to(device)] = v_pick["q16"][r_ids].to(device, non_blocking=True).float()
                q32_hi[p_pos.to(device)] = v_pick["q32_hi"][r_ids].to(device, non_blocking=True).float()
                y16[p_pos.to(device)] = v_pick["y16"][r_ids].to(device, non_blocking=True).float()
            q32_hi_aug = q32_hi + 0.05 * torch.randn_like(q32_hi)
            q32_hi_aug = q32_hi_aug - q32_hi_aug.mean(dim=(2, 3), keepdim=True)
        else:
            q32_hi_aug = None

        q16_aug = q16 + 0.06 * torch.randn_like(q16)
        q16_aug = q16_aug - q16_aug.mean(dim=(2, 3), keepdim=True)
        y16_aug, y16_clean_m = _blur_or_drop_y16(y16, drop_prob=0.40)

        ds = torch.rand(batch_size, device=device) < 0.15
        dc = torch.rand(batch_size, device=device) < 0.30
        lscale = torch.where(dc, torch.zeros(batch_size, device=device), 0.25 + 0.75 * torch.rand(batch_size, device=device))
        gamma_b = torch.where(
            torch.rand(batch_size, device=device) < 0.40,
            torch.ones(batch_size, device=device),
            0.15 + 0.85 * torch.rand(batch_size, device=device),
        )

        z0 = torch.randn_like(z1)
        tau = torch.rand(batch_size, device=device)
        zt = (1.0 - tau).view(batch_size, 1, 1, 1) * z0 + tau.view(batch_size, 1, 1, 1) * z1
        target_v = z1 - z0

        if ecore_tr is not None and priors_a_tr is not None and c64_tr is not None:
            # Incorporate Stage A3 analytic starting distribution for 35% of non-drop_content samples
            a_mask = (torch.rand(batch_size, device=device) < 0.35) & (~dc)
            if bool(a_mask.any()):
                a_pos = a_mask.nonzero().flatten()
                with torch.no_grad():
                    # Pair structure from painting i with random cross-painting start colors
                    perm_j = torch.randint(0, len(z_all), (len(a_pos),))
                    z_src_cross = z_all[perm_j].to(device).float()
                    c64_a = c64_tr[real_indices[a_pos.cpu()]].to(device).float()
                    q16_a = stiefel["train"]["q32"][real_indices[a_pos.cpu()]].to(device).float()
                    lam_a = 0.65 + 0.25 * torch.rand(len(a_pos), device=device)
                    ze_a = ecore_tr(
                        z_src_cross, c64_a, q16_a, tid[a_pos], lam_a,
                        raw_stats_tr, priors_ec_tr, noise=torch.randn_like(z_src_cross),
                    )
                    ze_mod = reconstruct_analytic_draft(ze_a, tid[a_pos], priors_a_tr, mode=analytic_mode)
                tau0_a = 0.60 + 0.20 * torch.rand(len(a_pos), device=device)
                s_a = torch.rand(len(a_pos), device=device)
                tau_br, zt_br, tv_br = sample_analytic_bridge_segment(z1[a_pos], z0[a_pos], ze_mod, tau0_a, s_a)
                tau[a_pos] = tau_br
                zt[a_pos] = zt_br
                target_v[a_pos] = tv_br

        phuber_acc, st_acc, pix_val, gn = run_pcore_train_step(
            model=model, opt=opt, ema=ema, params_list=params_list, swd_critic=swd_critic,
            w_luma=w_luma, b_luma=b_luma,
            z1=z1, zt=zt, target_v=target_v,
            y16=y16, y16_aug=y16_aug, y16_clean_m=y16_clean_m, q16_aug=q16_aug,
            tid=tid, u_in=u_in, tau=tau, ds=ds, dc=dc, lscale=lscale, gamma_b=gamma_b,
            micro=micro, accum=accum, swd_micro=swd_micro, step=step, seed_base=SEED_V34,
            use_adapter=use_adapter, q32_hi_aug=q32_hi_aug, device=device,
        )

        if step == 1 or step % 400 == 0 or step == target_steps:
            dt = (time.monotonic() - t0) / max(1, step - start_step)
            print(
                f"[{arm_name}] {step:4d}/{target_steps} phuber={phuber_acc:.4f} st={st_acc:.4f} "
                f"pix={pix_val:.4f} gn={float(gn):.2f} ({dt:.3f}s/st)",
                flush=True,
            )

        if smoke_steps == 0 and (step % 500 == 0 and step < target_steps):
            atomic_save({
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "cfg": cfg,
            }, resume_p)

    vram_gb = assert_vram_budget(limit_gb=vram_budget_gb, device=device, tag=arm_name)
    tot_gb = vram_total_gb(device)
    print(f"[{arm_name}] Peak VRAM = {vram_gb:.2f} / {tot_gb:.1f} GB ({(100.0 * vram_gb / tot_gb) if tot_gb > 0 else 0:.1f}%)", flush=True)
    elapsed_min = round((time.monotonic() - t0) / 60.0, 2)
    del model, opt, swd_critic
    gc.collect()
    torch.cuda.empty_cache()

    if smoke_steps > 0:
        del ema
        torch.cuda.empty_cache()
        return {"arm": arm_name, "cfg": cfg, "peak_vram_gb": round(vram_gb, 3), "elapsed_min": elapsed_min}

    atomic_save({
        "ema": ema.state_dict(),
        "cfg": cfg,
        "step": updates,
        "train_min": elapsed_min,
        "peak_vram_gb": round(vram_gb, 3),
        "nodino": True,
    }, ckpt_p)
    if resume_p.exists():
        resume_p.unlink()
    del ema
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "arm": arm_name,
        "ckpt": str(ckpt_p),
        "cfg": cfg,
        "train_min": elapsed_min,
        "peak_vram_gb": round(vram_gb, 3),
    }


@torch.no_grad()
def evaluate_v34_arm(
    arm_name: str,
    ckpt_path: Path,
    *,
    knob_list: tuple[float, ...] = (0.75, 0.85, 1.00, 1.25, 1.50),
    include_tau0: bool = False,
    eval_wrong: bool = True,
    analytic_mode: str = "A0",
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate a V34 checkpoint on full test750 via S10 + process-isolated referee."""
    sanitize_cuda_env()
    model, cfg = load_pcore_from_v34_ckpt(ckpt_path, device=device)
    style_mode = str(cfg.get("style_mode", "k8"))
    use_adapter = bool(cfg.get("use_adapter", False))

    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    q16_750 = stiefel["test750"]["q32"].float()
    cnx_gray_750 = stiefel["test750"]["cnx_gray"].float()
    inst_cache = build_or_load_style_instances(device=device) if style_mode != "k8" else None
    q32_hi_750 = build_or_load_q64_hi32_cache(device=device)["test750"]["q32_hi"].float() if use_adapter else None

    ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device)
    priors_ec = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device).eval().requires_grad_(False)
    ecore.load_state_dict(ec_sv["ema"], strict=True)
    priors_a = build_or_load_analytic_priors(priors_ec) if analytic_mode != "A0" else None

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    dec = load_s10_decoder(device)

    tasks, z_src, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    y16_750 = project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    c64_750 = ts["c_64"].float()
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    eval_specs: list[tuple[str, dict[str, Any], bool]] = []
    for kv in knob_list:
        kw = schedule_for_v34(kv)
        lbl = f"s{int(round(kv * 100)):03d}"
        eval_specs.append((lbl, kw, False))
    if include_tau0:
        eval_specs.append(("tau0", {"tau0": 0.0, "ws": 4.5, "wc": 1.8, "wsp": 1.35, "rescale_phi": 0.40, "gamma": 0.35, "lscale": 0.72, "draft_lambda": 0.75}, False))
    if eval_wrong:
        eval_specs.append(("s085_wrong", schedule_for_v34(0.85), True))

    draft_cache: dict[tuple[float, bool], torch.Tensor] = {}
    levels: dict[str, Any] = {}

    for lvl_tag, kw, wrong in eval_specs:
        full_tag = f"{arm_name}_{lvl_tag}"
        m_dir = OUT / "measured" / full_tag
        if not (m_dir / "summary_v34.json").exists():
            t0_v = float(kw["tau0"])
            lam = round(float(kw.get("draft_lambda", 0.75)), 3)
            if t0_v > 0.0 and (lam, wrong) not in draft_cache:
                chunks = []
                for st in range(0, len(tasks), 32):
                    sl = slice(st, min(st + 32, len(tasks)))
                    ti = pick_wrong_cross_tid(sid_750[sl], tid_750[sl]).to(device) if wrong else tid_750[sl].to(device)
                    ze = ecore(
                        z_src[sl].to(device).float(), c64_750[sl].to(device).float(),
                        q16_750[sl].to(device).float(), ti,
                        torch.full((sl.stop - sl.start,), lam, device=device),
                        raw_stats, priors_ec, noise=eps_750[sl].to(device).float(),
                    )
                    if priors_a is not None:
                        ze = reconstruct_analytic_draft(ze, ti, priors_a, mode=analytic_mode)
                    chunks.append(ze.cpu().half())
                draft_cache[(lam, wrong)] = torch.cat(chunks, 0)
            ze_pool = draft_cache.get((lam, wrong))

            writer = RawWriterV34(full_tag, dec=dec, device=device)
            kw_call = {k: v for k, v in kw.items() if k != "draft_lambda"}
            for st in range(0, len(tasks), 32):
                sl = slice(st, min(st + 32, len(tasks)))
                ti = pick_wrong_cross_tid(sid_750[sl], tid_750[sl]).to(device) if wrong else tid_750[sl].to(device)
                if style_mode == "k8":
                    u_st = match_cnx_k8_submode(cnx_gray_750[sl].to(device).float(), ti, cnx_bank)
                else:
                    assert inst_cache is not None
                    u_st = retrieve_continuous_style_u(cnx_gray_750[sl].to(device).float(), ti, inst_cache, cond_key=style_mode)
                ze = ze_pool[sl].to(device).float() if t0_v > 0.0 else None
                qh = q32_hi_750[sl].to(device).float() if q32_hi_750 is not None else None
                zo = integrate_v33_trajectory(
                    model, ze, eps_750[sl].to(device).float(),
                    y16_750[sl].to(device).float(), q16_750[sl].to(device).float(),
                    ti, u_st, steps=6, solver="dpm2m", q32_hi=qh, **kw_call,
                )
                writer.write_batch(zo, tasks[sl])
            writer.finalize(arm=full_tag, wrong=wrong, schedule=kw, analytic_mode=analytic_mode)

        res = measure_and_summarize_v34(full_tag, device=device)
        levels[lvl_tag] = res
        print(
            f"[V34 Eval] {full_tag:28s} | C={res['dino_c']:.4f} S={res['dino_s']:.4f} "
            f"Top5={res['dino_s_top5']:.4f} CSLS={res['csls_k10']:.4f} "
            f"Top1={res['csd_top1']*100:5.2f}% LP={res['lpips']:.4f} Same={res['same150_top1']*100:5.1f}%",
            flush=True,
        )

    del model, ecore, dec
    gc.collect()
    torch.cuda.empty_cache()
    return levels


@torch.no_grad()
def measure_style_diversity(
    arm_name: str,
    ckpt_path: Path,
    *,
    n_tasks: int = 50,
    device: str = "cuda",
) -> dict[str, Any]:
    """Diversity probe: keep source, noise, Q32, and E-Core draft fixed; vary only style vector u_A -> u_mid -> u_B."""
    sanitize_cuda_env()
    model, cfg = load_pcore_from_v34_ckpt(ckpt_path, device=device)
    style_mode = str(cfg.get("style_mode", "k8"))
    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    inst_cache = build_or_load_style_instances(device=device)

    ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device)
    priors_ec = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device).eval().requires_grad_(False)
    ecore.load_state_dict(ec_sv["ema"], strict=True)

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    dec = load_s10_decoder(device)
    tex_ext = MultiScaleTextureExtractor(device=device).eval()

    tasks, z_src, _ = tasks_and_sources()
    cross_idx = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]]
    stride = max(1, len(cross_idx) // n_tasks)
    pick = torch.tensor(cross_idx[::stride][:n_tasks], dtype=torch.long)

    zb = z_src[pick].to(device).float()
    c64_b = ts["c_64"][pick].to(device).float()
    q16_b = stiefel["test750"]["q32"][pick].to(device).float()
    y16_b = project_latent_y16_lcn(zb, v18["luma_proj"]["w"].to(device).float(), v18["luma_proj"]["b"].to(device).float())
    tid_b = torch.tensor([tasks[i]["target_style_id"] for i in pick.tolist()], dtype=torch.long, device=device)
    eps_full = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    eps_b = eps_full[pick].to(device).float()

    kw = {k: v for k, v in schedule_for_v34(0.85).items() if k != "draft_lambda"}
    ze_b = ecore(zb, c64_b, q16_b, tid_b, torch.full((len(pick),), 0.75, device=device), raw_stats, priors_ec, noise=eps_b)

    if style_mode == "k8":
        sub_417 = cnx_bank["sub_raw417"].to(device).float()
        # Pick the two farthest K=8 sub-schools per target style
        pair_u = []
        for s in range(5):
            norms = F.normalize(sub_417[s], dim=-1)
            flat = int((norms @ norms.T).argmin().item())
            ka, kb = divmod(flat, 8)
            pair_u.append((sub_417[s, ka], sub_417[s, kb]))
        u_a = torch.stack([pair_u[int(s)][0] for s in tid_b.tolist()], dim=0)
        u_b = torch.stack([pair_u[int(s)][1] for s in tid_b.tolist()], dim=0)
    else:
        ext_idx = inst_cache["extreme_pairs"]  # (5, 2)
        u_bank = inst_cache[style_mode].to(device).float()
        u_a = u_bank[ext_idx[tid_b.cpu(), 0]]
        u_b = u_bank[ext_idx[tid_b.cpu(), 1]]

    u_mid = 0.5 * u_a + 0.5 * u_b
    rgbs, texs = [], []
    for u_cond in (u_a, u_mid, u_b):
        zo = integrate_v33_trajectory(model, ze_b, eps_b, y16_b, q16_b, tid_b, u_cond, steps=6, solver="dpm2m", **kw)
        rgb = decode_s10(dec, zo, batch=25, output_cpu=False).clamp(0.0, 1.0)
        rgbs.append(rgb)
        texs.append(tex_ext(rgb))

    rgb_a, rgb_mid, rgb_b = rgbs
    tex_a, tex_mid, tex_b = texs
    pix_l1_ab = float((rgb_a - rgb_b).abs().mean().item())
    pix_l1_amid = float((rgb_a - rgb_mid).abs().mean().item())
    pix_l1_midb = float((rgb_mid - rgb_b).abs().mean().item())
    tex_cos_ab = float(F.cosine_similarity(tex_a, tex_b, dim=-1).mean().item())
    mean_rgb_shift = float((rgb_a.mean(dim=(2, 3)) - rgb_b.mean(dim=(2, 3))).abs().mean().item())
    midpoint_ratio = pix_l1_amid / max(pix_l1_ab, 1e-6)

    del model, ecore, dec, tex_ext
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "arm": arm_name,
        "style_mode": style_mode,
        "n_tasks": int(len(pick)),
        "pix_l1_uA_vs_uB": round(pix_l1_ab, 5),
        "pix_l1_uA_vs_uMid": round(pix_l1_amid, 5),
        "pix_l1_uMid_vs_uB": round(pix_l1_midb, 5),
        "midpoint_l1_ratio": round(midpoint_ratio, 4),
        "mean_rgb_color_shift": round(mean_rgb_shift, 5),
        "cnx_tex_cos_uA_vs_uB": round(tex_cos_ab, 5),
    }
