"""CLEAVE V33 P2 (~5.5h budget, Core Experiment): Strong-Style Generation -> Semantic Content Reconstruction.

SemanticControlAdapter (~6.9M params, zero-init) injects ONLY at 16x16 & 32x32 of frozen V32 P-Core (zero 64x64 skip!).
Time sampling: t ~ 0.5 * U(0, 1) + 0.5 * U(0, 0.65) to focus on low-retention structure reconstruction.

Three arms (3500 steps, effective batch 128, lr=3e-4, Pseudo-Huber delta=0.05):
  - S0: 16x16 Q16 upsampled to 32x32 (parameter-count & training control)
  - S1: 16x16 Q16 + true 512px 32x32 Q32_hi
  - S2: S1 + 30% style-perturbed condition training Q(T_style(x)) -> z_real
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import coarse_y16_lcn, project_latent_y16_lcn
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
from exp.v32_nodino.cnx_layout import extract_cnx_q32, match_cnx_k8_submode
from exp.v33_semantic_control import (
    OUT,
    S085_C10_KW,
    SEED_V33,
    V32_ECORE_CKPT,
    V32_PCORE_CKPT,
    RawWriterV33,
    measure_and_summarize,
    paired_ci_vs_baseline,
)
from exp.v33_semantic_control.assets import build_or_load_q64_hi32_cache, build_or_load_s2_pert_cache
from exp.v33_semantic_control.models import (
    PCoreWithSemanticControl,
    SemanticControlAdapter,
    integrate_v33_trajectory,
)
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc


def sample_p2_tau(n: int, device: str) -> torch.Tensor:
    """t ~ 0.5 * p_V32(t) + 0.5 * U(0, 0.65), where p_V32 = U(0, 1)."""
    u_full = torch.rand(n, device=device)
    u_low = 0.65 * torch.rand(n, device=device)
    choose_low = torch.rand(n, device=device) < 0.5
    return torch.where(choose_low, u_low, u_full)


@torch.no_grad()
def apply_style_perturbation_rgb(rgb01: torch.Tensor) -> torch.Tensor:
    """Apply T_style(x): random brightness curve, local contrast/shading tilt, and surface texture smoothing/noise."""
    B, _, H, W = rgb01.shape
    dev = rgb01.device
    x = rgb01.float().clamp(1e-4, 1.0)

    # 1. Brightness gamma curve + mild S-curve contrast
    gamma = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.60, 1.65)
    x = x.pow(gamma)
    gain = torch.empty(B, 1, 1, 1, device=dev).uniform_(1.2, 3.2)
    s_curve = 0.5 + 0.5 * torch.tanh(gain * (x - 0.5)) / torch.tanh(0.5 * gain)
    mix = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.0, 0.6)
    x = (1.0 - mix) * x + mix * s_curve

    # 2. Local contrast / smooth illumination field
    yy = torch.linspace(-1.0, 1.0, H, device=dev).view(1, 1, H, 1)
    xx = torch.linspace(-1.0, 1.0, W, device=dev).view(1, 1, 1, W)
    ax = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.25, 0.25)
    ay = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.25, 0.25)
    c_scale = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.78, 1.22)
    x = (x - 0.5) * c_scale + 0.5 + 0.15 * (ax * xx + ay * yy)

    # 3. Surface texture smoothing (50%) or mild grain perturbation (50%)
    sm_mask = (torch.rand(B, device=dev) < 0.5).view(B, 1, 1, 1)
    x_blur = gaussian_blur_3x3(gaussian_blur_3x3(x.clamp(0.0, 1.0)))
    x_grain = (x + 0.035 * torch.randn_like(x)).clamp(0.0, 1.0)
    return torch.where(sm_mask, x_blur, x_grain).clamp(0.0, 1.0)


def _load_frozen_v32_pcore(device: str = "cuda") -> PCoreSLFUNet:
    pc_sv = torch.load(V32_PCORE_CKPT, map_location="cpu", weights_only=False)
    pcore = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=True, normalize_u=True).to(device)
    pcore.load_state_dict(pc_sv["ema"], strict=True)
    pcore.eval().requires_grad_(False)
    return pcore


def p2_schedule_for_tau0(tau0: float) -> tuple[str, dict]:
    """Unified inference schedule across tau0 in {0.00, 0.35, 0.50, 0.65, 0.75, 0.79}."""
    t = round(float(tau0), 2)
    if abs(t - 0.79) < 0.015:
        return "tau079", dict(S085_C10_KW)
    if t <= 0.001:
        return "tau000", default_knob_schedule(1.50, use_gamma=True)
    # Map tau0 monotonically onto V32's calibrated curve:
    # tau0=0.75 -> s~0.89, tau0=0.65 -> s~0.99, tau0=0.50 -> s~1.16, tau0=0.35 -> s~1.31
    if t >= 0.635:
        alpha = (0.790 - t) / (0.790 - 0.635)
        k_lo, k_hi = S085_C10_KW, default_knob_schedule(1.00, use_gamma=True)
    else:
        alpha = (0.635 - t) / 0.635
        k_lo, k_hi = default_knob_schedule(1.00, use_gamma=True), default_knob_schedule(1.50, use_gamma=True)
    alpha = max(0.0, min(1.0, alpha))
    kw = {
        k: round((1.0 - alpha) * float(k_lo[k]) + alpha * float(k_hi[k]), 4 if k == "tau0" else 3)
        for k in S085_C10_KW
    }
    kw["tau0"] = float(tau0)
    return f"tau{int(round(t * 100)):03d}", kw


@torch.no_grad()
def evaluate_p2_arm(
    arm_name: str,
    ckpt_p: Path,
    mode: str,
    *,
    use_q64: bool = False,
    tau0_list: tuple[float, ...] = (0.00, 0.35, 0.50, 0.65, 0.75, 0.79),
    steps_list: tuple[int, ...] = (6,),
    eval_wrong: bool = True,
    device: str = "cuda",
) -> dict:
    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]

    pcore = _load_frozen_v32_pcore(device=device)
    sv = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    d_q16 = int(sv.get("d_q16", 64 if use_q64 else 32))
    ctrl = SemanticControlAdapter(d_q16=d_q16, d_q32=32).to(device)
    ctrl.load_state_dict(sv["ctrl_ema"], strict=True)
    ctrl.eval().requires_grad_(False)
    model = PCoreWithSemanticControl(pcore, ctrl).to(device).eval().requires_grad_(False)

    ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device).float()
    priors = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device).eval().requires_grad_(False)
    ecore.load_state_dict(ec_sv["ema"], strict=True)

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    dec = load_s10_decoder(device)
    tasks, z_src, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    y16_750 = project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    c64_750 = ts["c_64"].float()
    q16_750 = stiefel["test750"]["q32"].float()
    if use_q64:
        q16_in_750 = torch.cat([q16_750, q64_cache["test750"]["q_b"].float()], dim=1)
    else:
        q16_in_750 = q16_750
    cnx_gray_750 = stiefel["test750"]["cnx_gray"].float()
    if mode == "upsample16":
        q32_hi_750 = F.interpolate(q16_750, size=(32, 32), mode="bilinear", align_corners=False)
    else:
        q32_hi_750 = q64_cache["test750"]["q32_hi"].float()

    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    draft_cache: dict[tuple[float, bool], torch.Tensor] = {}
    out = {}

    eval_specs: list[tuple[str, float, dict, int, bool]] = []
    for n_st in steps_list:
        sfx_st = "" if n_st == 6 else f"_st{n_st}"
        for t0_v in tau0_list:
            lbl, kw = p2_schedule_for_tau0(t0_v)
            eval_specs.append((f"{lbl}{sfx_st}", t0_v, kw, n_st, False))
    if eval_wrong:
        _, kw_79 = p2_schedule_for_tau0(0.79)
        eval_specs.append(("tau079_wrong", 0.79, kw_79, 6, True))

    for lvl_tag, t0_v, kw, n_st, wrong in eval_specs:
        full_tag = f"{arm_name}_{lvl_tag}"
        m_dir = OUT / "measured" / full_tag
        if not (m_dir / "summary_v33.json").exists():
            lam = round(float(kw["draft_lambda"]), 3)
            if t0_v > 0.0 and (lam, wrong) not in draft_cache:
                chunks = []
                for st in range(0, len(tasks), 32):
                    sl = slice(st, min(st + 32, len(tasks)))
                    ti = pick_wrong_cross_tid(sid_750[sl], tid_750[sl]).to(device) if wrong else tid_750[sl].to(device)
                    chunks.append(ecore(
                        z_src[sl].to(device).float(), c64_750[sl].to(device).float(),
                        q16_750[sl].to(device).float(), ti,
                        torch.full((sl.stop - sl.start,), lam, device=device),
                        raw_stats, priors, noise=eps_750[sl].to(device).float(),
                    ).cpu().half())
                draft_cache[(lam, wrong)] = torch.cat(chunks, 0)
            ze_pool = draft_cache.get((lam, wrong))

            writer = RawWriterV33(full_tag, dec=dec, device=device)
            kw_call = {k: v for k, v in kw.items() if k != "draft_lambda"}
            for st in range(0, len(tasks), 32):
                sl = slice(st, min(st + 32, len(tasks)))
                ti = pick_wrong_cross_tid(sid_750[sl], tid_750[sl]).to(device) if wrong else tid_750[sl].to(device)
                u_st = match_cnx_k8_submode(cnx_gray_750[sl].to(device).float(), ti, cnx_bank)
                ze = ze_pool[sl].to(device).float() if t0_v > 0.0 else None
                zo = integrate_v33_trajectory(
                    model, ze, eps_750[sl].to(device).float(),
                    y16_750[sl].to(device).float(), q16_in_750[sl].to(device).float(),
                    ti, u_st, steps=n_st, solver="dpm2m",
                    q32_hi=q32_hi_750[sl].to(device).float(), **kw_call,
                )
                writer.write_batch(zo, tasks[sl])
            writer.finalize(arm=full_tag, tau0=t0_v, steps=n_st, wrong=wrong, schedule=kw)

        res = measure_and_summarize(full_tag, device=device)
        out[lvl_tag] = res
        print(
            f"[P2 Eval] {full_tag:28s} | C={res['dino_c']:.4f} S={res['dino_s']:.4f} Top5={res['dino_s_top5']:.4f} "
            f"CSLS={res['csls_k10']:.4f} Top1={res['csd_top1']*100:5.2f}% LP={res['lpips']:.4f} Same={res['same150_top1']*100:5.1f}%",
            flush=True,
        )

    del model, pcore, ctrl, ecore, dec
    gc.collect()
    torch.cuda.empty_cache()
    return out


def train_p2_arm(
    arm_name: str,
    mode: str,
    *,
    style_perturb_prob: float = 0.0,
    use_q64: bool = False,
    updates: int = 3500,
    lr: float = 3.0e-4,
    batch_size: int = 128,
    seed_offset: int = 0,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict:
    """Train SemanticControlAdapter arm (S0 upsample16, S1 true 32x32, S2 S1+style_perturb)."""
    sanitize_cuda_env()
    seed_all(SEED_V33 + 200 + seed_offset)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    run_dir = OUT / "p2" / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_p = run_dir / f"{arm_name}_s{updates}.pt"
    rep_p = run_dir / "arm_report.json"
    if rep_p.exists() and ckpt_p.exists() and smoke_steps == 0:
        print(f"[{arm_name}] Reusing completed P2 arm.", flush=True)
        return read_json(rep_p)

    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    w32_st = stiefel["W_stiefel_cnx"].float()
    wb32_st = q64_cache["W_b32"].float()
    whi32_st = q64_cache["W_hi32"].float()

    q16_all = stiefel["train"]["q32"]
    if use_q64:
        q16_all = torch.cat([q16_all, q64_cache["train"]["q_b"]], dim=1)
    if mode == "upsample16":
        q32_hi_all = F.interpolate(stiefel["train"]["q32"].float(), size=(32, 32), mode="bilinear", align_corners=False).half()
    else:
        q32_hi_all = q64_cache["train"]["q32_hi"]

    norm417_all = cnx_bank["norm_417_all"]
    sub_raw417 = cnx_bank["sub_raw417"]
    core_cls = [ci.cpu().long() for ci in cnx_bank["core_idx_per_style"]]

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
    pc_q32_hi = q32_hi_all[pc_ridx]

    swd_critic = _HighTEndpointSWDCritic(device=device)
    swd_critic.set_moment_scale(torch.cat([z_all[ci[:10]] for ci in core_cls], dim=0))
    pert_cache = build_or_load_s2_pert_cache(n_variants=2, device=device) if style_perturb_prob > 0.0 else None
    pert_vars = pert_cache["variants"] if pert_cache is not None else None

    pcore = _load_frozen_v32_pcore(device=device)
    d_q16 = 64 if use_q64 else 32
    ctrl = SemanticControlAdapter(d_q16=d_q16, d_q32=32).to(device)
    n_ctrl_params = ctrl.n_params()
    ctrl_ema = copy.deepcopy(ctrl).eval().requires_grad_(False)
    model = PCoreWithSemanticControl(pcore, ctrl).to(device)
    params_list = list(ctrl.parameters())
    opt = torch.optim.AdamW(params_list, lr=lr, betas=(0.9, 0.99), weight_decay=1.0e-4)

    micro, accum = tier_micro_accum(batch_size, default_micro_24gb=128, max_micro_12gb=32, max_micro_8gb=16, device=device)
    n_pc = int(round(batch_size * 0.15))
    n_self = batch_size - n_pc
    self_counts = _build_style_counts(n_self)
    target_steps = smoke_steps if smoke_steps > 0 else updates
    t0 = time.monotonic()

    for step in range(1, target_steps + 1):
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
        raw_idx_full = torch.cat([idx_self, pc_ridx[p_pick]], dim=0)

        z1_full = torch.cat([z_all[idx_self].float(), pc_zreal[p_pick].float()], dim=0)
        tid_full = torch.cat([sid_all[idx_self], pc_tid[p_pick]], dim=0)
        q16_full = torch.cat([q16_all[idx_self].float(), pc_q16[p_pick].float()], dim=0)
        q32_hi_full = torch.cat([q32_hi_all[idx_self].float(), pc_q32_hi[p_pick].float()], dim=0)
        y16_full = torch.cat([y16_all[idx_self].float(), pc_y16[p_pick].float()], dim=0)
        u_ex_full = torch.cat([norm417_all[idx_self].float(), norm417_all[pc_ridx[p_pick]].float()], dim=0)

        if pert_vars is not None and style_perturb_prob > 0.0:
            p_mask = torch.rand(batch_size) < style_perturb_prob
            if bool(p_mask.any()):
                p_pos = p_mask.nonzero().flatten()
                r_ids = raw_idx_full[p_pos]
                v_pick = pert_vars[int(step % len(pert_vars))]
                q16_p = v_pick["q16"][r_ids].float()
                if use_q64:
                    q16_p = torch.cat([q16_p, v_pick["q_b"][r_ids].float()], dim=1)
                q16_full[p_pos] = q16_p
                q32_hi_full[p_pos] = v_pick["q32_hi"][r_ids].float()
                y16_full[p_pos] = v_pick["y16"][r_ids].float()

        sub_k = torch.randint(0, 8, (batch_size,))
        u_sub_full = sub_raw417[tid_full, sub_k].float()
        u_in_full = torch.where(torch.rand(batch_size, 1) < 0.75, u_ex_full, u_sub_full)

        perm = torch.randperm(batch_size)
        z1_full, tid_full = z1_full[perm], tid_full[perm]
        q16_full, q32_hi_full = q16_full[perm], q32_hi_full[perm]
        y16_full, u_in_full = y16_full[perm], u_in_full[perm]

        opt.zero_grad(set_to_none=True)
        ph_acc, st_acc, pix_val = 0.0, 0.0, 0.0

        for a_i in range(accum):
            sl = slice(a_i * micro, (a_i + 1) * micro)
            mb = micro
            z1 = z1_full[sl].to(device, non_blocking=True)
            tid = tid_full[sl].to(device, non_blocking=True)
            q16 = q16_full[sl].to(device, non_blocking=True)
            q32_hi = q32_hi_full[sl].to(device, non_blocking=True)
            y16 = y16_full[sl].to(device, non_blocking=True)
            u_in = u_in_full[sl].to(device, non_blocking=True)

            q16_aug = q16 + 0.06 * torch.randn_like(q16)
            q16_aug = q16_aug - q16_aug.mean(dim=(2, 3), keepdim=True)
            q32_hi_aug = q32_hi + 0.06 * torch.randn_like(q32_hi)
            q32_hi_aug = q32_hi_aug - q32_hi_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug, y16_clean_m = _blur_or_drop_y16(y16, drop_prob=0.40)

            ds = torch.rand(mb, device=device) < 0.15
            dc = torch.rand(mb, device=device) < 0.30
            lscale = torch.where(dc, torch.zeros(mb, device=device), 0.25 + 0.75 * torch.rand(mb, device=device))
            gamma_b = torch.where(torch.rand(mb, device=device) < 0.40, torch.ones(mb, device=device), 0.15 + 0.85 * torch.rand(mb, device=device))

            z0 = torch.randn_like(z1)
            tau = sample_p2_tau(mb, device=device)
            zt = (1.0 - tau).view(mb, 1, 1, 1) * z0 + tau.view(mb, 1, 1, 1) * z1
            target_v = z1 - z0

            with bf16_autocast(device):
                pred_v = model(
                    zt, y16_aug, q16_aug, tid, u_in, tau,
                    q32_hi=q32_hi_aug, drop_style=ds, drop_content=dc, layout_scale=lscale, gamma=gamma_b,
                )
                phuber = _pseudo_huber_velocity(pred_v, target_v, delta=0.05)
                z1_hat = (zt + (1.0 - tau).view(mb, 1, 1, 1) * pred_v.float()).clamp(-5.5, 5.5)
                y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                w_st = (tau > 0.25).float() * (~dc).float() * y16_clean_m.float() * (gamma_b > 0.75).float()
                st_loss = ((y16_hat - y16).abs().mean(dim=(1, 2, 3)) * w_st).sum() / w_st.sum().clamp_min(1.0)
                loss_main = (phuber + 0.10 * st_loss) / float(accum)
            loss_main.backward()
            ph_acc += float(phuber.item()) / float(accum)
            st_acc += float(st_loss.item()) / float(accum)

            if step % 2 == 0 and a_i == 0:
                hi_idx = ((tau >= 0.60) & (tau <= 0.95) & (~dc)).nonzero().flatten()
                if len(hi_idx) >= 2:
                    pick = hi_idx[:6]
                    g_main = [p.grad.detach().clone() if p.grad is not None else None for p in params_list]
                    norm_main = math.sqrt(sum(float(g.square().sum().item()) for g in g_main if g is not None))
                    opt.zero_grad(set_to_none=True)

                    pg_s = torch.Generator(device=device).manual_seed(SEED_V33 + 9000 + step)
                    proj = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=pg_s), mode="reduced").Q for c in (64, 128, 256))
                    with bf16_autocast(device):
                        pv_hi = model(
                            zt[pick], y16_aug[pick], q16_aug[pick], tid[pick], u_in[pick], tau[pick],
                            q32_hi=q32_hi_aug[pick], drop_style=ds[pick], drop_content=dc[pick],
                            layout_scale=lscale[pick], gamma=gamma_b[pick],
                        )
                        z1_hi = (zt[pick] + (1.0 - tau[pick]).view(-1, 1, 1, 1) * pv_hi.float()).clamp(-5.5, 5.5)
                    l_pix = swd_critic.loss(z1_hi, z1[pick], proj)
                    pix_val = float(l_pix.item())
                    l_pix.backward()

                    norm_pix = math.sqrt(sum(float(p.grad.square().sum().item()) for p in params_list if p.grad is not None))
                    scale = min(2.0, (0.25 * norm_main) / max(norm_pix, 1e-6))
                    for p, gm in zip(params_list, g_main):
                        if p.grad is not None and gm is not None:
                            p.grad.mul_(scale).add_(gm)
                        elif gm is not None:
                            p.grad = gm

        gn = torch.nn.utils.clip_grad_norm_(params_list, 1.0, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ctrl_ema.parameters()), [p.detach() for p in params_list], 0.004)

        if step == 1 or step % 400 == 0 or step == target_steps:
            dt = (time.monotonic() - t0) / step
            print(f"[{arm_name}] {step:4d}/{target_steps} phuber={ph_acc:.4f} st={st_acc:.4f} pix={pix_val:.4f} gn={float(gn):.2f} ({dt:.3f}s/st)", flush=True)

        if smoke_steps == 0 and step % 500 == 0 and step < updates:
            atomic_save({"ctrl_ema": ctrl_ema.state_dict(), "step": step, "mode": mode, "d_q16": d_q16, "params": n_ctrl_params}, run_dir / f"{arm_name}_s{step}.pt")

    vram_gb = assert_vram_budget(device=device, tag=arm_name)
    elapsed_min = round((time.monotonic() - t0) / 60.0, 2)
    del model, pcore, ctrl, opt, swd_critic, pert_cache, pert_vars
    gc.collect()
    torch.cuda.empty_cache()

    if smoke_steps > 0:
        del ctrl_ema
        torch.cuda.empty_cache()
        return {"arm": arm_name, "mode": mode, "ctrl_params": n_ctrl_params, "peak_vram_gb": round(vram_gb, 3), "elapsed_min": elapsed_min}

    atomic_save({"ctrl_ema": ctrl_ema.state_dict(), "step": updates, "mode": mode, "d_q16": d_q16, "params": n_ctrl_params}, ckpt_p)
    del ctrl_ema
    gc.collect()
    torch.cuda.empty_cache()

    eval_out = evaluate_p2_arm(arm_name, ckpt_p, mode, use_q64=use_q64, device=device)
    rep = {
        "updated_utc": utc(),
        "arm": arm_name,
        "mode": mode,
        "style_perturb_prob": style_perturb_prob,
        "d_q16": d_q16,
        "ctrl_params": n_ctrl_params,
        "train_min": elapsed_min,
        "peak_vram_gb": round(vram_gb, 3),
        "ckpt": str(ckpt_p),
        "levels": eval_out,
    }
    save_json(rep_p, rep)
    return rep


def run_p2(*, use_q64: bool = False, smoke: bool = False, device: str = "cuda") -> dict:
    sanitize_cuda_env()
    t0 = time.monotonic()
    arms_spec = [
        ("P2_S0_upsample16", "upsample16", 0.0),
        ("P2_S1_hi32", "true32", 0.0),
        ("P2_S2_hi32_pert30", "true32", 0.30),
    ]
    if smoke:
        smoke_res = {}
        for name, md, pp in arms_spec:
            smoke_res[name] = train_p2_arm(name, md, style_perturb_prob=pp, use_q64=use_q64, smoke_steps=20, device=device)
        return {"smoke": smoke_res}

    arm_reports = {}
    for name, md, pp in arms_spec:
        arm_reports[name] = train_p2_arm(name, md, style_perturb_prob=pp, use_q64=use_q64, updates=3500, device=device)

    # Pick winning P2 arm by Pareto push at core working zone (tau0 in {0.50, 0.65, 0.75, 0.79})
    def _score_arm(rep: dict) -> float:
        lv = rep["levels"]
        # Balanced score rewarding C & S improvements in the C~0.50-0.65 frontier
        return sum(lv[k]["dino_c"] + 1.4 * lv[k]["dino_s"] + 0.15 * lv[k]["csd_top1"] for k in ("tau050", "tau065", "tau075", "tau079"))

    winner_name = max(arm_reports.keys(), key=lambda k: _score_arm(arm_reports[k]))
    winner_rep = arm_reports[winner_name]

    # Compare 8 and 12 integration steps on the winning arm
    winner_steps = evaluate_p2_arm(
        winner_name,
        Path(winner_rep["ckpt"]),
        winner_rep["mode"],
        use_q64=use_q64,
        tau0_list=(0.50, 0.65, 0.79),
        steps_list=(8, 12),
        eval_wrong=False,
        device=device,
    )

    ci_s2_vs_s0 = {
        k: paired_ci_vs_baseline(OUT / f"measured/{winner_name}_{k}/eval.json", OUT / f"measured/P2_S0_upsample16_{k}/eval.json")
        for k in ("tau050", "tau065", "tau079")
    }

    report = {
        "updated_utc": utc(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "arms": arm_reports,
        "winner_arm": winner_name,
        "winner_steps_8_12": winner_steps,
        "ci_winner_vs_s0": ci_s2_vs_s0,
    }
    save_json(OUT / "p2_report.json", report)
    print(f"[P2 DONE] in {report['elapsed_s']:.1f}s | Winner={winner_name}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="CLEAVE V33 P2 Semantic Control Adapter")
    ap.add_argument("--use-q64", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_p2(use_q64=args.use_q64, smoke=args.smoke, device=args.device)


if __name__ == "__main__":
    main()
