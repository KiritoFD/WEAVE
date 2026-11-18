"""WEAVE V20 Master Sequencer: Surpassing StyleGallery on the (DINO-C, DINO-S / Affinity) Pareto Frontier.

Stages ("小步快跑", each stage emits full isolated 750-task referee metrics + 512x512 PNGs):
  --canary : 15s end-to-end verification of offline ConvNeXt Patch-SWD, SLFUNet, and PRISMFlashNet
  --stage1 : Vivid Sub-School Bank + Multi-Scale ConvNeXt/RGB Patch-SWD Polish on T4 (~3.5 min)
  --stage2 : Coarse-16 Semantic-Layout Flow (SLFUNet) + T4 Flow-Bridge + Patch-SWD Pro (~15 min)
  --stage3 : High-DINO-S 1-Step Chord Student (PRISM-1 v3) & Zero-DINO PRISM-Flash (~5 min)
  --all    : Run stages 1 -> 2 -> 3 sequentially
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    EVAL_SEED,
    OUT,
    ROOT,
    STYLES,
    RawWriterV20,
    autocast,
    export_raw_pngs,
    pick_wrong_cross_tid,
    read_json,
    run_measure_v20,
    sanitize_cuda_env,
    save_json,
    seed_all,
    tier_micro_accum,
    utc,
)
from .slf_flow import (
    SLFUNet,
    coarse_y16_lcn,
    finetune_slf_core,
    integrate_slf_batch,
    integrate_slf_core_batch,
    polish_with_submode_patchswd,
    project_latent_y16_lcn,
    train_slf_flow,
)
from .submode_patchswd import (
    MultiScalePatchSWDExtractor,
    build_or_load_core_knn_bank,
    build_or_load_vivid_submode_bank,
    load_raw_latents_750,
    match_scene_submode,
)

DEFAULT_POOL = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")


def run_canary(device: str = "cuda") -> None:
    """Fast 15-second end-to-end verification of all V20 components on GPU."""
    from src.model.prism import PRISMFlashNet
    from src.utils import decode_s10, load_s10_decoder

    sanitize_cuda_env()
    print("[V20 Canary] 1/3 Checking offline ConvNeXt-Tiny + MultiScalePatchSWDExtractor through S10-BF16...", flush=True)
    dec_fp16 = load_s10_decoder(device)
    dec_bf16 = copy.deepcopy(dec_fp16).to(device=device, dtype=torch.bfloat16).eval()
    for p in dec_bf16.parameters():
        p.requires_grad_(False)

    swd_ext = MultiScalePatchSWDExtractor(n_slices=64).to(device).eval()
    z_ref = torch.randn(2, 4, 64, 64, device=device) * 0.5
    with torch.no_grad():
        rgb_ref = decode_s10(dec_fp16, z_ref, batch=2, output_cpu=False).clamp(0.0, 1.0)
        ref_tok = swd_ext.extract_tokens(rgb_ref)

    zr = (torch.randn(2, 4, 64, 64, device=device) * 0.5).requires_grad_(True)
    rgb = decode_s10(dec_bf16, zr, batch=2, dtype=torch.bfloat16, enable_grad=True, output_cpu=False)
    loss_swd = swd_ext.swd_loss(rgb, ref_tok).sum()
    loss_swd.backward()
    assert zr.grad is not None and torch.isfinite(zr.grad).all() and float(zr.grad.abs().max()) > 0.0
    del dec_fp16, dec_bf16, swd_ext
    torch.cuda.empty_cache()

    print("[V20 Canary] 2/3 Checking Coarse-16 SLFUNet forward/backward & RK2 Dual-CFG sampler...", flush=True)
    slf = SLFUNet(base_dim=144, u_dim=128).to(device)
    zt = torch.randn(2, 4, 64, 64, device=device)
    y16 = torch.randn(2, 1, 16, 16, device=device)
    d16 = torch.randn(2, 384, 16, 16, device=device)
    sid = torch.tensor([0, 4], device=device)
    uv = F.normalize(torch.randn(2, 128, device=device), dim=-1)
    tau = torch.tensor([0.3, 0.7], device=device)
    with autocast(device):
        v = slf(zt, y16, d16, sid, uv, tau)
    v.float().square().mean().backward()
    slf.eval()
    z_ode = integrate_slf_batch(slf, zt, y16, d16, sid, uv, tau0=0.35, steps=2, w_style=3.5, w_content=1.2)
    assert z_ode.shape == (2, 4, 64, 64) and torch.isfinite(z_ode).all()
    n_slf = slf.n_params()
    del slf
    torch.cuda.empty_cache()

    print("[V20 Canary] 3/3 Checking Zero-DINO PRISMFlashNet & exact zero-chord self-identity...", flush=True)
    flash = PRISMFlashNet(width=96, chord_dim=128).to(device).eval()
    with torch.no_grad():
        flash.out_conv.weight.normal_(0, 0.02)
        z_in = torch.randn(2, 4, 64, 64, device=device) * 0.6
        # Self-reference z_ref == z_in -> chord == 0 -> exact z_in!
        z_self = flash(z_in, z_ref=z_in)
        self_err = float((z_self - z_in).abs().max().item())
        assert self_err < 1e-6, f"Zero-chord identity failed: {self_err}"
        z_cross = flash(z_in, tid=torch.tensor([1, 4], device=device))
        assert float((z_cross - z_in).abs().max().item()) > 0.0
    print(
        f"V20_CANARY_ALL_OK (SLFUNet={n_slf/1e6:.2f}M, PRISMFlashNet={flash.n_params()/1e6:.2f}M)",
        flush=True,
    )


def ensure_t4_latents_750(bundle: dict, device: str = "cuda") -> torch.Tensor:
    """Load V19_T4_sweet_vgg_nosid 750 latents from runs/v19, or compute once if absent."""
    v19_t4 = ROOT / "runs/v19/raw/V19_T4_sweet_vgg_nosid"
    cache_t4 = OUT / "cache/t4_latents_750.pt"
    if cache_t4.exists():
        return torch.load(cache_t4, map_location="cpu", weights_only=False)
    if (v19_t4 / "meta.json").exists():
        print("[V20] Loading cached V19_T4_sweet_vgg_nosid latents from runs/v19...", flush=True)
        z_t4 = load_raw_latents_750(v19_t4)
        cache_t4.parent.mkdir(parents=True, exist_ok=True)
        torch.save(z_t4.half(), cache_t4)
        return z_t4.float()

    from exp.v19.aniso_teacher import AnisotropicS10Teacher, run_aniso_teacher_arm
    teacher = AnisotropicS10Teacher(bundle, device=device)
    run_aniso_teacher_arm(
        "V19_T4_sweet_vgg_nosid", teacher, mode="s10_vgg",
        alpha_c=0.18, steps=2, rho_y_edge=0.09, rho_y_low=0.15, rho_c=0.22, lcn_budget=0.025,
    )
    del teacher
    torch.cuda.empty_cache()
    z_t4 = load_raw_latents_750(v19_t4)
    cache_t4.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z_t4.half(), cache_t4)
    return z_t4.float()


def get_test750_y16_and_d16(bundle: dict) -> tuple[torch.Tensor, torch.Tensor]:
    from exp.v18 import tasks_and_sources
    _, z_src_750, _ = tasks_and_sources()
    ts = bundle["test_struct"]
    d16_750 = ts["dino_16"].float()
    if "y_64" in ts:
        y16_750 = coarse_y16_lcn(ts["y_64"].float())
    else:
        w_luma = bundle["v18_cache"]["luma_proj"]["w"].float()
        b_luma = bundle["v18_cache"]["luma_proj"]["b"].float()
        y16_750 = project_latent_y16_lcn(z_src_750.float(), w_luma, b_luma)
    return y16_750, d16_750


def run_stage1_patchswd(bundle: dict, bank: dict, device: str = "cuda") -> list[str]:
    """Stage 1 (~3.5 min): Polish V19_T4 with Vivid Scene-Matched Sub-School Patch-SWD."""
    from exp.v18 import tasks_and_sources
    from exp.v19.aniso_teacher import AnisotropicS10Teacher

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4_750 = ensure_t4_latents_750(bundle, device=device).float()
    _, d16_750 = get_test750_y16_and_d16(bundle)
    fib_750 = bundle["fnd"]["test750_fib"].float()

    arms = [
        ("V20_A1_t4_patchswd", dict(steps=3, rho_y_edge=0.11, rho_y_low=0.20, rho_c=0.34, lcn_allow=0.042)),
        ("V20_A2_t4_patchswd_wide", dict(steps=4, rho_y_edge=0.14, rho_y_low=0.26, rho_c=0.44, lcn_allow=0.060)),
    ]

    need_any = any(not (OUT / f"raw/{a}/meta.json").exists() for a, _ in arms)
    if need_any:
        teacher = AnisotropicS10Teacher(bundle, device=device)
        swd_ext = MultiScalePatchSWDExtractor(n_slices=96).to(device).eval()

        for arm_name, kw in arms:
            if (OUT / f"raw/{arm_name}/meta.json").exists():
                continue
            t0 = time.time()
            wr = RawWriterV20(arm_name, dec=teacher.dec_fp16, device=device)
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
                tid_p = torch.tensor([t["target_style_id"] for t in grp], device=device)
                same = sid_p == tid_p
                z_out = polish_with_submode_patchswd(
                    z_t4_750[sl], z_src_750[sl], fib_750[sl], d16_750[sl], tid_p,
                    teacher, swd_ext, bank, same_mask=same, **kw,
                )
                wr.write_batch(z_out, grp)
            meta = wr.finalize(total_wall_sec=round(time.time() - t0, 2), config=kw)
            print(f"[V20 Stage1] {arm_name} generated in {meta['total_wall_sec']}s", flush=True)

        del teacher, swd_ext
        torch.cuda.empty_cache()

    done = []
    for arm_name, _ in arms:
        if not (OUT / f"measured/{arm_name}/summary_v20.json").exists():
            run_measure_v20(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        done.append(arm_name)
    return done


def run_stage2_slf_flow(bundle: dict, bank: dict, updates: int = 5000, device: str = "cuda") -> list[str]:
    """Stage 2 (~14 min): Train Coarse-16 SLFUNet & sweep tau0 in {0.0, 0.35, 0.50} + Patch-SWD Pro."""
    from exp.v18 import tasks_and_sources
    from exp.v19.aniso_teacher import AnisotropicS10Teacher

    micro, accum = tier_micro_accum(128, default_micro_24gb=128, max_micro_12gb=32, device=device)
    ckpt_p = train_slf_flow(bundle, bank, updates=updates, micro=micro, accum=accum, device=device)

    slf = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
    slf.load_state_dict(torch.load(ckpt_p, map_location=device, weights_only=False)["ema"])
    for p in slf.parameters():
        p.requires_grad_(False)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4_750 = ensure_t4_latents_750(bundle, device=device).float()
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    fib_750 = bundle["fnd"]["test750_fib"].float()

    # Deterministic standard normal noise bank for Test750 (100% reproducible across tau0!)
    g_eval = torch.Generator().manual_seed(EVAL_SEED)
    eps_750 = torch.randn(z_src_750.shape, generator=g_eval)

    # (arm_name, tau0, steps, w_style, w_content, c8_blend, is_wrong)
    flow_arms = [
        ("V20_B0_slf_coarse8_tau0", 0.00, 18, 4.2, 1.10, 1.00, False),
        ("V20_B1c_slf_soft8_tau0", 0.00, 18, 4.0, 1.18, 0.50, False),
        ("V20_B1_slf_tau0_ws35", 0.00, 18, 3.5, 1.25, 0.00, False),
        ("V20_B1b_slf_tau0_ws45", 0.00, 18, 4.5, 1.20, 0.00, False),
        ("V20_B2_slf_bridge35_ws35", 0.35, 14, 3.8, 1.18, 0.25, False),
        ("V20_B3_slf_bridge50_ws38", 0.50, 12, 3.8, 1.15, 0.00, False),
        ("V20_B4_slf_wrong", 0.35, 14, 3.8, 1.18, 0.25, True),
    ]

    from src.utils import load_s10_decoder
    dec_s10 = load_s10_decoder(device)
    infer_bs = 25 if micro >= 64 else 10
    for arm_name, tau0, steps, ws, wc, c8_b, is_wrong in flow_arms:
        if (OUT / f"raw/{arm_name}/meta.json").exists():
            continue
        t0 = time.time()
        wr = RawWriterV20(arm_name, dec=dec_s10, device=device)
        for st in range(0, 750, infer_bs):
            sl = slice(st, min(st + infer_bs, 750))
            grp = tasks[sl]
            zs = z_src_750[sl].to(device).float()
            zt4 = z_t4_750[sl].to(device).float()
            ep = eps_750[sl].to(device).float()
            y16 = y16_750[sl].to(device).float()
            d16 = d16_750[sl].to(device).float()
            if c8_b > 0.0:
                d8 = F.avg_pool2d(d16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
                d8 = d8 - d8.mean(dim=(2, 3), keepdim=True)
                y8 = F.avg_pool2d(y16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
                d16_in = (1.0 - c8_b) * d16 + c8_b * d8
                d16_in = d16_in - d16_in.mean(dim=(2, 3), keepdim=True)
                y16_in = (1.0 - c8_b) * y16 + c8_b * y8
            else:
                d16_in, y16_in = d16, y16

            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_true = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_true
            tid_u = pick_wrong_cross_tid(sid_p, tid_true) if is_wrong else tid_true

            _, u_viv = match_scene_submode(d16, tid_u, bank)
            if tau0 <= 0.0:
                z_start = ep
            else:
                z_anc = zs if is_wrong else zt4
                z_start = (1.0 - tau0) * ep + tau0 * z_anc

            z_gen = integrate_slf_batch(
                slf, z_start, y16_in, d16_in, tid_u, u_viv,
                tau0=tau0, steps=steps, w_style=ws, w_content=wc,
            )
            z_out = torch.where(same.view(-1, 1, 1, 1), zs, z_gen)
            wr.write_batch(z_out, grp)

        meta = wr.finalize(
            total_wall_sec=round(time.time() - t0, 2),
            tau0=tau0, steps=steps, w_style=ws, w_content=wc, c8_blend=c8_b, is_wrong=is_wrong,
        )
        print(f"[V20 Stage2] {arm_name} generated in {meta['total_wall_sec']}s", flush=True)

    del slf, dec_s10
    torch.cuda.empty_cache()

    done = []
    for arm_name, *_ in flow_arms:
        if not (OUT / f"measured/{arm_name}/summary_v20.json").exists():
            run_measure_v20(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        done.append(arm_name)

    # Stage 2b: Polish the two strongest generative SLF arms (tau0=0.0 and bridge35) with 2-step Submode Patch-SWD + VGG!
    combo_arms = [
        ("V20_C0_soft8_swd_pro", "V20_B1c_slf_soft8_tau0", dict(steps=2, rho_y_edge=0.14, rho_y_low=0.24, rho_c=0.36, lcn_allow=0.115)),
        ("V20_C1_slf_tau0_swd_pro", "V20_B1_slf_tau0_ws35", dict(steps=2, rho_y_edge=0.14, rho_y_low=0.24, rho_c=0.36, lcn_allow=0.095)),
        ("V20_C2_bridge35_swd_pro", "V20_B2_slf_bridge35_ws35", dict(steps=2, rho_y_edge=0.12, rho_y_low=0.22, rho_c=0.34, lcn_allow=0.068)),
        ("V20_C3_bridge50_swd_pro", "V20_B3_slf_bridge50_ws38", dict(steps=2, rho_y_edge=0.11, rho_y_low=0.20, rho_c=0.32, lcn_allow=0.052)),
    ]
    if any(not (OUT / f"raw/{a}/meta.json").exists() for a, _, _ in combo_arms):
        teacher = AnisotropicS10Teacher(bundle, device=device)
        swd_ext = MultiScalePatchSWDExtractor(n_slices=96).to(device).eval()
        for arm_name, base_arm, kw in combo_arms:
            if (OUT / f"raw/{arm_name}/meta.json").exists():
                continue
            z_base = load_raw_latents_750(OUT / f"raw/{base_arm}")
            t0 = time.time()
            wr = RawWriterV20(arm_name, dec=teacher.dec_fp16, device=device)
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
                tid_p = torch.tensor([t["target_style_id"] for t in grp], device=device)
                same = sid_p == tid_p
                z_out = polish_with_submode_patchswd(
                    z_base[sl], z_src_750[sl], fib_750[sl], d16_750[sl], tid_p,
                    teacher, swd_ext, bank, same_mask=same, **kw,
                )
                wr.write_batch(z_out, grp)
            meta = wr.finalize(total_wall_sec=round(time.time() - t0, 2), base_arm=base_arm, config=kw)
            print(f"[V20 Stage2b] {arm_name} generated in {meta['total_wall_sec']}s", flush=True)
        del teacher, swd_ext
        torch.cuda.empty_cache()

    for arm_name, _, _ in combo_arms:
        if not (OUT / f"measured/{arm_name}/summary_v20.json").exists():
            run_measure_v20(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        done.append(arm_name)
    return done


def run_stage2c_core_slf(bundle: dict, core_bank: dict, updates: int = 1800, device: str = "cuda") -> list[str]:
    """Stage 2c (~16 min): Fine-tune SLFUNet on blind Intra-Train Core-kNN manifold & surpass StyleGallery."""
    from exp.v18 import tasks_and_sources
    from exp.v19.aniso_teacher import AnisotropicS10Teacher
    from src.utils import load_s10_decoder

    micro, accum = tier_micro_accum(128, default_micro_24gb=128, max_micro_12gb=32, device=device)
    base_ckpt = OUT / "slf_flow/slf_final.pt"
    if not base_ckpt.exists():
        base_ckpt = train_slf_flow(bundle, core_bank, updates=5000, micro=micro, accum=accum, device=device)
    core_ckpt = finetune_slf_core(bundle, core_bank, base_ckpt, updates=updates, micro=micro, accum=accum, device=device)

    slf = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
    slf.load_state_dict(torch.load(core_ckpt, map_location=device, weights_only=False)["ema"], strict=True)
    for p in slf.parameters():
        p.requires_grad_(False)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4_750 = ensure_t4_latents_750(bundle, device=device).float()
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    fib_750 = bundle["fnd"]["test750_fib"].float()

    g_eval = torch.Generator().manual_seed(EVAL_SEED)
    eps_750 = torch.randn(z_src_750.shape, generator=g_eval)

    # (arm_name, kw_dict, is_wrong)
    core_arms = [
        (
            "V20_D1_core_slf_hiS",
            dict(tau_hi=0.34, steps=16, ws_fig=3.9, wsp_fig_late=5.4, wsp_abs=5.8, wc_fig=1.50, wc_abs=0.50, lscale_abs=0.22),
            False,
        ),
        (
            "V20_D2_core_slf_bal",
            dict(tau_hi=0.44, steps=16, ws_fig=3.8, wsp_fig_late=4.8, wsp_abs=5.4, wc_fig=1.60, wc_abs=0.60, lscale_abs=0.28),
            False,
        ),
        (
            "V20_D3_core_slf_hiC",
            dict(tau_hi=0.54, steps=14, ws_fig=3.7, wsp_fig_late=4.2, wsp_abs=4.8, wc_fig=1.65, wc_abs=0.72, lscale_abs=0.36),
            False,
        ),
        (
            "V20_D5_core_slf_c64",
            dict(tau_min=0.28, tau_hi=0.66, steps=10, ws_fig=4.2, wsp_fig_late=2.4, wsp_abs=5.2, wc_fig=1.85, wc_abs=0.80, lscale_abs=0.34),
            False,
        ),
        (
            "V20_D6_core_slf_c69",
            dict(tau_min=0.40, tau_hi=0.74, steps=8, ws_fig=4.4, wsp_fig_late=1.8, wsp_abs=4.8, wc_fig=1.95, wc_abs=1.00, lscale_abs=0.44),
            False,
        ),
        (
            "V20_D7_core_slf_domSG",
            dict(tau_min=0.00, tau_hi=0.72, steps=10, ws_fig=4.3, wsp_fig_late=2.0, wsp_abs=5.6, wc_fig=1.90, wc_abs=0.52, lscale_abs=0.22),
            False,
        ),
        (
            "V20_D4_core_slf_wrong",
            dict(tau_hi=0.44, steps=16, ws_fig=3.8, wsp_fig_late=4.8, wsp_abs=5.4, wc_fig=1.60, wc_abs=0.60, lscale_abs=0.28),
            True,
        ),
    ]

    dec_s10 = load_s10_decoder(device)
    infer_bs = 25 if micro >= 64 else 10
    for arm_name, kw, is_wrong in core_arms:
        if (OUT / f"raw/{arm_name}/meta.json").exists():
            continue
        t0 = time.time()
        wr = RawWriterV20(arm_name, dec=dec_s10, device=device)
        for st in range(0, 750, infer_bs):
            sl = slice(st, min(st + infer_bs, 750))
            grp = tasks[sl]
            zs = z_src_750[sl].to(device).float()
            zt4 = z_t4_750[sl].to(device).float()
            ep = eps_750[sl].to(device).float()
            y16 = y16_750[sl].to(device).float()
            d16 = d16_750[sl].to(device).float()

            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_true = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_true
            tid_u = pick_wrong_cross_tid(sid_p, tid_true) if is_wrong else tid_true
            z_anc = zs if is_wrong else zt4

            z_gen = integrate_slf_core_batch(slf, z_anc, ep, y16, d16, tid_u, core_bank, **kw)
            z_out = torch.where(same.view(-1, 1, 1, 1), zs, z_gen)
            wr.write_batch(z_out, grp)

        meta = wr.finalize(total_wall_sec=round(time.time() - t0, 2), is_wrong=is_wrong, **kw)
        print(f"[V20 Stage2c] {arm_name} generated in {meta['total_wall_sec']}s ({round(meta['total_wall_sec']*1000/750, 1)} ms/img)", flush=True)

    del slf, dec_s10
    torch.cuda.empty_cache()

    done = []
    for arm_name, _, _ in core_arms:
        if not (OUT / f"measured/{arm_name}/summary_v20.json").exists():
            run_measure_v20(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        done.append(arm_name)

    # 2-step Core-kNN Patch-SWD + VGG Pro polish on D1 and D2!
    pro_arms = [
        ("V20_E1_core_hiS_swd_pro", "V20_D1_core_slf_hiS", dict(steps=2, rho_y_edge=0.12, rho_y_low=0.22, rho_c=0.34, lcn_allow=0.095)),
        ("V20_E2_core_bal_swd_pro", "V20_D2_core_slf_bal", dict(steps=2, rho_y_edge=0.11, rho_y_low=0.20, rho_c=0.32, lcn_allow=0.075)),
    ]
    if any(not (OUT / f"raw/{a}/meta.json").exists() for a, _, _ in pro_arms):
        teacher = AnisotropicS10Teacher(bundle, device=device)
        swd_ext = MultiScalePatchSWDExtractor(n_slices=96).to(device).eval()
        for arm_name, base_arm, kw in pro_arms:
            if (OUT / f"raw/{arm_name}/meta.json").exists():
                continue
            z_base = load_raw_latents_750(OUT / f"raw/{base_arm}")
            t0 = time.time()
            wr = RawWriterV20(arm_name, dec=teacher.dec_fp16, device=device)
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
                tid_p = torch.tensor([t["target_style_id"] for t in grp], device=device)
                same = sid_p == tid_p
                z_out = polish_with_submode_patchswd(
                    z_base[sl], z_src_750[sl], fib_750[sl], d16_750[sl], tid_p,
                    teacher, swd_ext, core_bank, same_mask=same, **kw,
                )
                wr.write_batch(z_out, grp)
            meta = wr.finalize(total_wall_sec=round(time.time() - t0, 2), base_arm=base_arm, config=kw)
            print(f"[V20 Stage2c] {arm_name} generated in {meta['total_wall_sec']}s", flush=True)
        del teacher, swd_ext
        torch.cuda.empty_cache()

    for arm_name, _, _ in pro_arms:
        if not (OUT / f"measured/{arm_name}/summary_v20.json").exists():
            run_measure_v20(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        done.append(arm_name)
    return done


def run_stage3_fast_students(bundle: dict, bank: dict, device: str = "cuda") -> list[str]:
    """Stage 3 (~5 min): Distill SLF+PatchSWD teacher into 1-step PRISM-1 v3 & Zero-DINO PRISM-Flash (<1ms latent!)."""
    from exp.v18 import tasks_and_sources
    from exp.v19.fiber_duality import BalancedVGGCriticHead, StyleFiberEncoder, UnnormalizedVGGFeatureExtractor
    from src.model.prism import PRISMFlashNet, extract_latent_multiscale_stats
    from src.utils import decode_s10, load_s10_decoder

    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    priors = fnd["priors"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].float()

    # Prefer Core-kNN SLF v2 checkpoint for Stage 3 fast student distillation!
    slf_ckpt = OUT / "slf_flow/slf_core_v2.pt"
    if not slf_ckpt.exists():
        slf_ckpt = OUT / "slf_flow/slf_final.pt"
    if not slf_ckpt.exists():
        print("[V20 Stage3] Training SLFUNet first...", flush=True)
        micro, accum = tier_micro_accum(128, default_micro_24gb=128, max_micro_12gb=32, device=device)
        slf_ckpt = train_slf_flow(bundle, bank, updates=5000, micro=micro, accum=accum, device=device)

    pairs_p = OUT / "cache/flash_distill_pairs_core.pt"
    if pairs_p.exists():
        pairs = torch.load(pairs_p, map_location="cpu", weights_only=False)
    else:
        print("[V20 Stage3] Generating 2,500 clean cross-style + self-paired Core-SLF targets on train_4400...", flush=True)
        seed_all(20261121)
        slf = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
        slf.load_state_dict(torch.load(slf_ckpt, map_location=device, weights_only=False)["ema"], strict=True)
        from exp.v19.fiber_duality import apply_latent_chroma_shift

        N_tot = len(tr["z"])
        n_pairs = 2500
        src_idx = torch.randint(0, N_tot, (n_pairs,))
        sid_p = tr["sid"][src_idx].long()
        tid_p = torch.randint(0, 5, (n_pairs,))
        # Force 15% self-pairs, 85% cross-pairs
        force_same = torch.rand(n_pairs) < 0.15
        tid_p = torch.where(force_same, sid_p, (sid_p + torch.randint(1, 5, (n_pairs,))) % 5)

        y16_tr = coarse_y16_lcn(tr["y_64"].float())
        z_targets = []
        bs = 50
        for st in range(0, n_pairs, bs):
            sl = slice(st, min(st + bs, n_pairs))
            ii = src_idx[sl]
            zs = tr["z"][ii].to(device).float()
            ti = tid_p[sl].to(device)
            si = sid_p[sl].to(device)
            d16 = tr["dino_16"][ii].to(device).float()
            y16 = y16_tr[ii].to(device).float()
            z_chr = apply_latent_chroma_shift(zs, ti, priors, alpha=0.25)
            if "mid_dsd" in bank:
                z_gen = integrate_slf_core_batch(
                    slf, z_chr, torch.randn_like(zs), y16, d16, ti, bank,
                    tau_hi=0.45, steps=10,
                )
            else:
                _, uv = match_scene_submode(d16, ti, bank)
                z_st = 0.55 * torch.randn_like(zs) + 0.45 * z_chr
                z_gen = integrate_slf_batch(slf, z_st, y16, d16, ti, uv, tau0=0.45, steps=10, w_style=3.6, w_content=1.2)
            z_gen = torch.where((si == ti).view(-1, 1, 1, 1), zs, z_gen)
            z_targets.append(z_gen.cpu().half())
        del slf
        torch.cuda.empty_cache()
        pairs = {"src_idx": src_idx, "sid": sid_p, "tid": tid_p, "z_star": torch.cat(z_targets, dim=0)}
        pairs_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pairs, pairs_p)

    flash_ckpt = OUT / "students/prism_flash_v20.pt"
    flash_ckpt.parent.mkdir(parents=True, exist_ok=True)
    flash = PRISMFlashNet(width=96, chord_dim=128).to(device)

    if flash_ckpt.exists():
        flash.load_state_dict(torch.load(flash_ckpt, map_location=device, weights_only=False)["model"])
        flash.eval()
    else:
        seed_all(20261122)
        # 1. Populate geometric buffers + Haar HF stds from clean train_4400
        from src.utils.structure import dwt2_haar
        flash.w_luma.copy_(w_luma.to(device))
        flash.b_luma.copy_(b_luma.to(device))
        flash.P_Y.copy_(priors["P_Y"].to(device))
        flash.P_C.copy_(priors["P_C"].to(device))
        flash.cls_mu_c.copy_(priors["mu_c"].to(device))
        flash.cls_std_c.copy_(priors["std_c"].to(device))

        with torch.no_grad():
            tr_z_f = tr["z"].float()
            tr_sid = tr["sid"].long()
            for s in range(5):
                zs_c = tr_z_f[tr_sid == s].to(device)
                _, lh, hl, hh = dwt2_haar(zs_c)
                hf = torch.cat([lh, hl, hh], dim=1)
                flash.cls_hf_std[s].copy_(hf.std(dim=(0, 2, 3), keepdim=True, unbiased=False).squeeze(0))
            st_all = torch.cat([extract_latent_multiscale_stats(tr_z_f[i:i+256].to(device)).cpu() for i in range(0, len(tr_z_f), 256)], 0)
            flash.fiber_stem.s_mean.copy_(st_all.mean(0, keepdim=True).to(device))
            flash.fiber_stem.s_std.copy_(st_all.std(0, keepdim=True).clamp_min(1e-4).to(device))

        # 2. Pretrain LatentStyleFiberStem (700 steps, ~8s) & SelfAttentiveLatentLayoutStem (700 steps, ~10s)
        fiber_t = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
        fiber_t.load_state_dict(fnd["fiber_state"])
        with torch.no_grad():
            u_teacher_tr = fiber_t.encode(fnd["tr_fib"].to(device).float())
            flash.fiber_stem.protos.data.copy_(fiber_t.class_barycenters())

        opt_f = torch.optim.AdamW(flash.fiber_stem.parameters(), lr=2e-3, weight_decay=1e-3)
        for _ in range(700):
            idx = torch.randint(0, len(tr_z_f), (128,))
            zb = tr_z_f[idx].to(device)
            yb = tr_sid[idx].to(device)
            ut = u_teacher_tr[idx]
            u_hat = flash.fiber_stem.encode(zb)
            lg = flash.fiber_stem.logits(zb)
            loss = F.cross_entropy(lg, yb) + 1.5 * (1.0 - (u_hat * ut).sum(dim=-1)).mean()
            opt_f.zero_grad(set_to_none=True)
            loss.backward()
            opt_f.step()

        with torch.no_grad():
            ho_z = bundle["v18_cache"]["holdout"]["z"].float().to(device)
            ho_y = bundle["v18_cache"]["holdout"]["sid"].long().to(device)
            ho_acc = float((flash.fiber_stem.logits(ho_z).argmax(1) == ho_y).float().mean().item())
        print(f"[V20 Flash] LatentStyleFiberStem holdout 5-way acc = {ho_acc*100:.2f}% (0 DINO!)", flush=True)

        from src.utils.structure import latent_to_structure_64
        opt_l = torch.optim.AdamW(flash.layout_stem.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(700):
            idx = torch.randint(0, len(tr_z_f), (96,))
            zb = tr_z_f[idx].to(device)
            d16_t = tr["dino_16"][idx].to(device).float()
            w = flash.w_luma.view(1, 4, 1, 1)
            y64 = ((zb * w).sum(1, keepdim=True) + flash.b_luma).clamp(0.0, 1.0)
            c64 = latent_to_structure_64(zb, flash.w_luma, flash.b_luma)
            q16 = flash.layout_stem.forward_tokens(y64, c64)
            pred_d16 = flash.layout_stem.distill_proj(q16)
            loss = F.smooth_l1_loss(pred_d16, d16_t)
            opt_l.zero_grad(set_to_none=True)
            loss.backward()
            opt_l.step()

        # 3. End-to-End Student Training: SLF Distillation + S10-BF16 ConvNeXt Patch-SWD + VGG Calibrated Margin!
        dec_fp16 = load_s10_decoder(device)
        dec_bf16 = copy.deepcopy(dec_fp16).to(device=device, dtype=torch.bfloat16).eval()
        for p in dec_bf16.parameters():
            p.requires_grad_(False)
        swd_ext = MultiScalePatchSWDExtractor(n_slices=64).to(device).eval()
        vgg_ext = UnnormalizedVGGFeatureExtractor(gram_rank=48).to(device).eval()
        vgg_head = BalancedVGGCriticHead(in_dim=fnd["vgg_in_dim"], num_styles=5).to(device).eval()
        vgg_head.load_state_dict(fnd["vgg_head_state"])
        for p in vgg_head.parameters():
            p.requires_grad_(False)

        ema = copy.deepcopy(flash).eval()
        for p in ema.parameters():
            p.requires_grad_(False)

        opt = torch.optim.AdamW(flash.parameters(), lr=8e-4, weight_decay=1e-4)
        n_steps = 1800
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=5e-5)
        t0 = time.time()
        N_p = len(pairs["src_idx"])

        for step in range(1, n_steps + 1):
            idx = torch.randint(0, N_p, (32,))
            s_ii = pairs["src_idx"][idx]
            zs = tr_z_f[s_ii].to(device)
            z_star = pairs["z_star"][idx].to(device).float()
            ti = pairs["tid"][idx].to(device)
            si = pairs["sid"][idx].to(device)
            same = si == ti
            d16_s = tr["dino_16"][s_ii].to(device).float()

            z_pred = flash(zs, tid=ti, same_mask=same)
            l_lat = F.smooth_l1_loss(z_pred, z_star)

            # Every step, backprop Patch-SWD + VGG calibrated margin on a 6-sample cross-style sub-slice!
            cross_m = (~same).nonzero(as_tuple=True)[0][:6]
            if len(cross_m) > 0:
                zp_sub = z_pred[cross_m]
                ti_sub = ti[cross_m]
                ex_idx, _ = match_scene_submode(d16_s[cross_m], ti_sub, bank)
                ex_cpu = ex_idx.detach().cpu()
                rt_sub = {k: v[ex_cpu].to(device).float() for k, v in bank["ex_tokens"].items()}
                rgb_sub = decode_s10(dec_bf16, zp_sub, batch=6, dtype=torch.bfloat16, enable_grad=True, output_cpu=False)
                swd_l = swd_ext.swd_loss(rgb_sub, rt_sub).mean()
                q_vgg = vgg_head(vgg_ext(rgb_sub, multi_scale=True), calibrated=True)
                oh = F.one_hot(ti_sub, 5).bool()
                m_vgg = (q_vgg.gather(1, ti_sub[:, None]).squeeze(1) - q_vgg.masked_fill(oh, -1e9).max(1).values).mean()
                l_pix = 0.45 * swd_l + 0.06 * F.relu(2.5 - m_vgg)
            else:
                l_pix = torch.tensor(0.0, device=device)

            loss = l_lat + l_pix
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(flash.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pw in zip(ema.parameters(), flash.parameters()):
                    pe.mul_(0.996).add_(pw, alpha=0.004)
            if step == 1 or step % 300 == 0 or step == n_steps:
                print(f"[V20 Flash] {step:4d}/{n_steps} lat={l_lat.item():.4f} pix={l_pix.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

        del dec_fp16, dec_bf16, swd_ext, vgg_ext, vgg_head
        torch.cuda.empty_cache()
        flash = ema.eval()
        torch.save({"model": flash.state_dict(), "params": flash.n_params(), "ho_acc": ho_acc}, flash_ckpt)

    # Evaluate PRISMFlashNet on Test750 + measure true GPU wall-clock latency!
    tasks, z_src_750, _ = tasks_and_sources()
    flash_arms = [
        ("V20_F1_prism_flash", 1.0, True, False),
        ("V20_F2_prism_flash_l125", 1.25, True, False),
        ("V20_F3_prism_flash_blind", 1.0, False, False),
        ("V20_F4_prism_flash_wrong", 1.0, True, True),
    ]
    dec_s10 = load_s10_decoder(device)
    for arm_name, lam, use_same_gate, is_wrong in flash_arms:
        if (OUT / f"raw/{arm_name}/meta.json").exists():
            continue
        wr = RawWriterV20(arm_name, dec=dec_s10, device=device)
        lat_chunks = []
        torch.cuda.synchronize()
        t0 = time.time()
        for st in range(0, 750, 50):
            sl = slice(st, min(st + 50, 750))
            grp = tasks[sl]
            zs = z_src_750[sl].to(device).float()
            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_t
            tid_u = pick_wrong_cross_tid(sid_p, tid_t) if is_wrong else tid_t
            with torch.no_grad():
                if not use_same_gate:
                    # Pure blind source-label-free soft gate via LatentStyleFiberStem!
                    u_s = flash.fiber_stem.encode(zs)
                    u_t = flash.fiber_stem.class_barycenters()[tid_u]
                    pred_same = (u_s * u_t).sum(-1) > 0.78
                    z_out = flash(zs, tid=tid_u, strength=lam, same_mask=pred_same)
                else:
                    z_out = flash(zs, tid=tid_u, strength=lam, same_mask=same)
            lat_chunks.append((z_out.detach(), grp))
        torch.cuda.synchronize()
        wall = time.time() - t0
        for z_out, grp in lat_chunks:
            wr.write_batch(z_out, grp)
        meta = wr.finalize(total_wall_sec=round(wall, 3), ms_per_img_latent=round(wall * 1000.0 / 750.0, 3))
        print(f"[V20 Flash] {arm_name}: 750 imgs in {wall:.2f}s ({meta['ms_per_img_latent']} ms/img latent!)", flush=True)
    del dec_s10
    torch.cuda.empty_cache()

    done = []
    for arm_name, *_ in flash_arms:
        if not (OUT / f"measured/{arm_name}/summary_v20.json").exists():
            run_measure_v20(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        done.append(arm_name)
    return done


def run_stage4_anchored_flash(bundle: dict, core_bank: dict, device: str = "cuda") -> list[str]:
    """Stage 4 (~4.5 min): Anchored PRISM-Flash v2 (0.6ms latent / ~7ms E2E) with LCN Lock + Patch-SWD."""
    from exp.v18 import tasks_and_sources
    from exp.v19.aniso_teacher import latent_lcn_cost
    from exp.v19.fiber_duality import BalancedVGGCriticHead, StyleFiberEncoder, UnnormalizedVGGFeatureExtractor
    from exp.v19.routed_student import build_or_load_v19_distill_pairs
    from src.model.prism import PRISMFlashNet, extract_latent_multiscale_stats
    from src.utils import decode_s10, load_s10_decoder
    from src.utils.structure import dwt2_haar, latent_to_structure_64

    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    priors = fnd["priors"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].float()

    pairs = build_or_load_v19_distill_pairs(bundle, teacher_mode="hybrid", n_per_style=100, device=device)

    flash_v2_ckpt = OUT / "students/prism_flash_v2_anchored.pt"
    flash_v2_ckpt.parent.mkdir(parents=True, exist_ok=True)
    flash = PRISMFlashNet(width=96, chord_dim=128).to(device)

    if flash_v2_ckpt.exists():
        flash.load_state_dict(torch.load(flash_v2_ckpt, map_location=device, weights_only=False)["model"])
        flash.eval()
    else:
        seed_all(20261125)
        flash.w_luma.copy_(w_luma.to(device))
        flash.b_luma.copy_(b_luma.to(device))
        flash.P_Y.copy_(priors["P_Y"].to(device))
        flash.P_C.copy_(priors["P_C"].to(device))
        flash.cls_mu_c.copy_(priors["mu_c"].to(device))
        flash.cls_std_c.copy_(priors["std_c"].to(device))

        with torch.no_grad():
            tr_z_f = tr["z"].float()
            tr_sid = tr["sid"].long()
            for s in range(5):
                zs_c = tr_z_f[tr_sid == s].to(device)
                _, lh, hl, hh = dwt2_haar(zs_c)
                hf = torch.cat([lh, hl, hh], dim=1)
                flash.cls_hf_std[s].copy_(hf.std(dim=(0, 2, 3), keepdim=True, unbiased=False).squeeze(0))
            st_all = torch.cat([extract_latent_multiscale_stats(tr_z_f[i:i+256].to(device)).cpu() for i in range(0, len(tr_z_f), 256)], 0)
            flash.fiber_stem.s_mean.copy_(st_all.mean(0, keepdim=True).to(device))
            flash.fiber_stem.s_std.copy_(st_all.std(0, keepdim=True).clamp_min(1e-4).to(device))

        fiber_t = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
        fiber_t.load_state_dict(fnd["fiber_state"])
        with torch.no_grad():
            u_teacher_tr = fiber_t.encode(fnd["tr_fib"].to(device).float())
            flash.fiber_stem.protos.data.copy_(fiber_t.class_barycenters())

        opt_f = torch.optim.AdamW(flash.fiber_stem.parameters(), lr=2e-3, weight_decay=1e-3)
        for _ in range(700):
            idx = torch.randint(0, len(tr_z_f), (128,))
            zb = tr_z_f[idx].to(device)
            yb = tr_sid[idx].to(device)
            ut = u_teacher_tr[idx]
            u_hat = flash.fiber_stem.encode(zb)
            lg = flash.fiber_stem.logits(zb)
            loss = F.cross_entropy(lg, yb) + 1.5 * (1.0 - (u_hat * ut).sum(dim=-1)).mean()
            opt_f.zero_grad(set_to_none=True)
            loss.backward()
            opt_f.step()

        opt_l = torch.optim.AdamW(flash.layout_stem.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(700):
            idx = torch.randint(0, len(tr_z_f), (96,))
            zb = tr_z_f[idx].to(device)
            d16_t = tr["dino_16"][idx].to(device).float()
            w = flash.w_luma.view(1, 4, 1, 1)
            y64 = ((zb * w).sum(1, keepdim=True) + flash.b_luma).clamp(0.0, 1.0)
            c64 = latent_to_structure_64(zb, flash.w_luma, flash.b_luma)
            q16 = flash.layout_stem.forward_tokens(y64, c64)
            loss = F.smooth_l1_loss(flash.layout_stem.distill_proj(q16), d16_t)
            opt_l.zero_grad(set_to_none=True)
            loss.backward()
            opt_l.step()

        dec_fp16 = load_s10_decoder(device)
        dec_bf16 = copy.deepcopy(dec_fp16).to(device=device, dtype=torch.bfloat16).eval()
        for p in dec_bf16.parameters():
            p.requires_grad_(False)
        swd_ext = MultiScalePatchSWDExtractor(n_slices=64).to(device).eval()
        vgg_ext = UnnormalizedVGGFeatureExtractor(gram_rank=48).to(device).eval()
        vgg_head = BalancedVGGCriticHead(in_dim=fnd["vgg_in_dim"], num_styles=5).to(device).eval()
        vgg_head.load_state_dict(fnd["vgg_head_state"])
        for p in vgg_head.parameters():
            p.requires_grad_(False)

        ema = copy.deepcopy(flash).eval()
        for p in ema.parameters():
            p.requires_grad_(False)

        for p in flash.fiber_stem.parameters():
            p.requires_grad_(False)

        opt = torch.optim.AdamW([p for p in flash.parameters() if p.requires_grad], lr=9e-4, weight_decay=1e-4)
        n_steps = 1600
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=6e-5)
        N_p = len(pairs["z_src"])
        t0 = time.time()

        for step in range(1, n_steps + 1):
            idx = torch.randint(0, N_p, (48,))
            zs = pairs["z_src"][idx].to(device).float()
            z_star = pairs["z_star"][idx].to(device).float()
            ti = pairs["tid"][idx].to(device)
            si = pairs["sid"][idx].to(device)
            same = si == ti
            d16_s = pairs["dino_16"][idx].to(device).float()

            z_pred = flash(zs, tid=ti, same_mask=same)
            l_lat = F.smooth_l1_loss(z_pred, z_star)
            l_lcn = latent_lcn_cost(z_pred, zs, flash.w_luma, flash.b_luma).mean()

            cross_m = (~same).nonzero(as_tuple=True)[0][:6]
            if len(cross_m) > 0:
                zp_sub = z_pred[cross_m]
                ti_sub = ti[cross_m]
                ex_idx, _ = match_scene_submode(d16_s[cross_m], ti_sub, core_bank)
                ex_cpu = ex_idx.detach().cpu()
                rt_sub = {k: v[ex_cpu].to(device).float() for k, v in core_bank["ex_tokens"].items()}
                rgb_sub = decode_s10(dec_bf16, zp_sub, batch=6, dtype=torch.bfloat16, enable_grad=True, output_cpu=False)
                swd_l = swd_ext.swd_loss(rgb_sub, rt_sub).mean()
                q_vgg = vgg_head(vgg_ext(rgb_sub, multi_scale=True), calibrated=True)
                oh = F.one_hot(ti_sub, 5).bool()
                m_vgg = (q_vgg.gather(1, ti_sub[:, None]).squeeze(1) - q_vgg.masked_fill(oh, -1e9).max(1).values).mean()
                l_pix = 0.08 * swd_l + 0.025 * F.relu(2.2 - m_vgg)
            else:
                l_pix = torch.tensor(0.0, device=device)

            loss = l_lat + 3.5 * l_lcn + l_pix
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(flash.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pw in zip(ema.parameters(), flash.parameters()):
                    pe.mul_(0.996).add_(pw, alpha=0.004)
            if step == 1 or step % 400 == 0 or step == n_steps:
                print(f"[V20 FlashV2] {step:4d}/{n_steps} lat={l_lat.item():.4f} lcn={l_lcn.item():.4f} pix={l_pix.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

        del dec_fp16, dec_bf16, swd_ext, vgg_ext, vgg_head
        torch.cuda.empty_cache()
        flash = ema.eval()
        torch.save({"model": flash.state_dict(), "params": flash.n_params()}, flash_v2_ckpt)

    tasks, z_src_750, _ = tasks_and_sources()
    dec_s10 = load_s10_decoder(device)

    with torch.no_grad():
        zb_w = z_src_750[:25].to(device).float()
        tb_w = torch.zeros(25, dtype=torch.long, device=device)
        for _ in range(3):
            decode_s10(dec_s10, flash(zb_w, tid=tb_w), batch=25, output_cpu=False)
        torch.cuda.synchronize()
        t_e2e = time.perf_counter()
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            zs = z_src_750[sl].to(device).float()
            tid_t = torch.tensor([t["target_style_id"] for t in tasks[sl]], device=device)
            z_o = flash(zs, tid=tid_t)
            decode_s10(dec_s10, z_o, batch=25, output_cpu=False)
        torch.cuda.synchronize()
        e2e_ms = round((time.perf_counter() - t_e2e) * 1000.0 / 750.0, 2)
    print(f"[V20 FlashV2] End-to-End (PRISMFlashNet + S10) on 4090: {e2e_ms} ms/img ({round(1000.0/e2e_ms, 1)} img/s)!", flush=True)

    flash_arms = [
        ("V20_F5_prism_flash_v2", 1.0, False),
        ("V20_F6_prism_flash_v2_l125", 1.25, False),
        ("V20_F7_prism_flash_v2_wrong", 1.0, True),
    ]
    for arm_name, lam, is_wrong in flash_arms:
        if (OUT / f"raw/{arm_name}/meta.json").exists():
            continue
        wr = RawWriterV20(arm_name, dec=dec_s10, device=device)
        lat_chunks = []
        torch.cuda.synchronize()
        t0 = time.time()
        for st in range(0, 750, 50):
            sl = slice(st, min(st + 50, 750))
            grp = tasks[sl]
            zs = z_src_750[sl].to(device).float()
            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_t
            tid_u = pick_wrong_cross_tid(sid_p, tid_t) if is_wrong else tid_t
            with torch.no_grad():
                z_out = flash(zs, tid=tid_u, strength=lam, same_mask=same)
            lat_chunks.append((z_out.detach(), grp))
        torch.cuda.synchronize()
        wall = time.time() - t0
        for z_out, grp in lat_chunks:
            wr.write_batch(z_out, grp)
        meta = wr.finalize(
            total_wall_sec=round(wall, 3),
            ms_per_img_latent=round(wall * 1000.0 / 750.0, 3),
            ms_per_img_e2e=e2e_ms,
        )
        print(f"[V20 FlashV2] {arm_name}: latent={meta['ms_per_img_latent']} ms/img, e2e={e2e_ms} ms/img", flush=True)

    del dec_s10
    torch.cuda.empty_cache()

    done = []
    for arm_name, *_ in flash_arms:
        if not (OUT / f"measured/{arm_name}/summary_v20.json").exists():
            run_measure_v20(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        done.append(arm_name)
    return done


def write_v20_summary() -> None:
    rows = []
    if (OUT / "measured").exists():
        for p in sorted((OUT / "measured").glob("*/summary_v20.json")):
            arm_nm = p.parent.name
            s = read_json(p)
            c = s["cross600"]
            sm = s["same150"]
            ms_img = None
            mp = OUT / "raw" / arm_nm / "meta.json"
            if mp.exists():
                m = read_json(mp)
                if m.get("ms_per_img_e2e") is not None:
                    ms_img = round(float(m["ms_per_img_e2e"]), 2)
                elif m.get("total_wall_sec") is not None:
                    ms_img = round(float(m["total_wall_sec"]) * 1000.0 / 750.0, 2)
            rows.append({
                "arm": arm_nm,
                "is_diagnostic": ("_wrong" in arm_nm) or arm_nm.startswith(("V20_E", "V20_F")),
                "csd_top1": round(c["csd_top1"], 5),
                "csd_margin": round(c["csd_margin"], 5) if c.get("csd_margin") is not None else None,
                "dino_s": round(c["dino_s"], 5),
                "clip_s": round(c["clip_s"], 5),
                "style_affinity": round(c["style_affinity"], 5),
                "dino_c": round(c["dino_c"], 5),
                "lpips": round(c["lpips"], 5),
                "same150_top1": round(sm["csd_top1"], 5),
                "same150_dino_s": round(sm["dino_s"], 5) if sm.get("dino_s") is not None else None,
                "same150_dino_c": round(sm["dino_c"], 5) if sm.get("dino_c") is not None else None,
                "ms_per_img": ms_img,
                "per_target_top1": {k: round(v["csd_top1"], 4) for k, v in s["per_target_cross120"].items()},
                "per_target_dino_s": {k: round(v["dino_s"], 4) for k, v in s["per_target_cross120"].items()},
            })
    save_json(OUT / "v20_master_summary.json", {"updated_utc": utc(), "arms": rows})
    print(f"[V20 Summary] Wrote {len(rows)} arms -> {OUT / 'v20_master_summary.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--stage2c", action="store_true")
    ap.add_argument("--stage3", action="store_true")
    ap.add_argument("--stage4", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slf-updates", type=int, default=5000)
    ap.add_argument("--core-updates", type=int, default=1800)
    args = ap.parse_args()

    sanitize_cuda_env()
    if args.canary:
        run_canary(device=args.device)
        return

    from exp.v19.fiber_duality import build_or_load_v19_foundation
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_vivid_submode_bank(bundle, k_sub=8, device=args.device)
    core_bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)

    if args.stage1 or args.all:
        run_stage1_patchswd(bundle, bank, device=args.device)
        write_v20_summary()
    if args.stage2 or args.all:
        run_stage2_slf_flow(bundle, bank, updates=args.slf_updates, device=args.device)
        write_v20_summary()
    if args.stage2c or args.all:
        run_stage2c_core_slf(bundle, core_bank, updates=args.core_updates, device=args.device)
        write_v20_summary()
    if args.stage3 or args.all:
        run_stage3_fast_students(bundle, core_bank, device=args.device)
        write_v20_summary()
    if args.stage4 or args.all:
        run_stage4_anchored_flash(bundle, core_bank, device=args.device)
        write_v20_summary()


if __name__ == "__main__":
    main()
