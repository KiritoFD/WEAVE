"""CLEAVE V33 P3 (~2h budget): Joint Winning Combination, 7000-step Training, Seed-1 Re-verification,
Full Causal Audit (Correct/Zero/Shuffled, Wrong-Target, Same150), Bootstrap CI, & Cold-Start Throughput.
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
from exp.v25.r14_e2e_benchmark import load_a0_encoder
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
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import extract_cnx_q32, match_cnx_k8_submode
from exp.v33_semantic_control import (
    OUT,
    S085_C10_KW,
    SEED_V33,
    V32_ECORE_CKPT,
    V32_OUT,
    V32_PCORE_CKPT,
    RawWriterV33,
    measure_and_summarize,
    paired_ci_vs_baseline,
)
from exp.v33_semantic_control.assets import build_or_load_q64_hi32_cache, build_or_load_routing_and_energy, build_or_load_s2_pert_cache, match_cnx_k8_csls
from exp.v33_semantic_control.models import (
    ConvNeXtMultiScaleExtractor,
    PCoreWithSemanticControl,
    SemanticControlAdapter,
    extract_cnx_q32_hi,
    integrate_v33_trajectory,
)
from exp.v33_semantic_control.p2_semantic_control import (
    p2_schedule_for_tau0,
    sample_p2_tau,
    train_p2_arm,
)
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import REPO as ROOT, bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc
from src.utils.structure import local_contrast_norm, rgb_to_gray, rgb_to_structure_64


def _v32_frontier_points() -> list[tuple[float, float]]:
    ledger = read_json(V32_OUT / "v32_final_ledger.json")["spectrum"]
    pts = []
    for k in ("ECore_NoDINO_l075", "PCore_NoDINO_s075_cal_c10", "PCore_NoDINO_s085_c10", "PCore_NoDINO_s093_cal_c10", "PCore_NoDINO_s100", "PCore_NoDINO_s125", "PCore_NoDINO_s150"):
        if k in ledger:
            pts.append((float(ledger[k]["dino_c"]), float(ledger[k]["dino_s"])))
    return sorted(pts, key=lambda x: x[0])


def _interp_v32_s_at_c(c_val: float, pts: list[tuple[float, float]]) -> float:
    if c_val <= pts[0][0]:
        return pts[0][1]
    if c_val >= pts[-1][0]:
        return pts[-1][1]
    for (c0, s0), (c1, s1) in zip(pts, pts[1:]):
        if c0 <= c_val <= c1:
            a = (c_val - c0) / max(c1 - c0, 1e-6)
            return (1.0 - a) * s0 + a * s1
    return pts[-1][1]


def check_p2_frontier_push(p2_rep: dict) -> dict:
    """Check whether the P2 winner pushes the (DINO-C, DINO-S) frontier above V32."""
    v32_pts = _v32_frontier_points()
    w_name = p2_rep["winner_arm"]
    lv = p2_rep["arms"][w_name]["levels"]
    deltas = {}
    for k, m in lv.items():
        if "wrong" in k:
            continue
        c_v, s_v = float(m["dino_c"]), float(m["dino_s"])
        s_base = _interp_v32_s_at_c(c_v, v32_pts)
        deltas[k] = {"dino_c": c_v, "dino_s": s_v, "v32_s_at_same_c": round(s_base, 4), "delta_s": round(s_v - s_base, 4)}
    max_push = max((v["delta_s"] for v in deltas.values()), default=-1.0)
    return {"winner": w_name, "max_delta_s_at_same_c": round(max_push, 4), "pushed_frontier": bool(max_push > 0.002), "per_point": deltas}


def _train_joint_finetune(
    arm_name: str,
    init_p2_ckpt: Path,
    mode: str,
    style_perturb_prob: float,
    *,
    use_q64: bool = False,
    updates: int = 3500,
    ctrl_lr: float = 9.0e-5,
    backbone_lr: float = 3.0e-5,
    batch_size: int = 128,
    seed_offset: int = 0,
    device: str = "cuda",
) -> Path:
    """Joint fine-tuning from 3500-step SemanticControl checkpoint with 0.3x ctrl LR & 0.1x backbone LR."""
    sanitize_cuda_env()
    seed_all(SEED_V33 + 500 + seed_offset)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    run_dir = OUT / "p3" / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_p = run_dir / f"{arm_name}_joint.pt"
    if ckpt_p.exists():
        print(f"[{arm_name}] Reusing completed joint checkpoint.", flush=True)
        return ckpt_p

    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    w32_st = stiefel["W_stiefel_cnx"].float()
    wb32_st = q64_cache["W_b32"].float()
    whi32_st = q64_cache["W_hi32"].float()

    q16_all = stiefel["train"]["q32"]
    if use_q64:
        q16_all = torch.cat([q16_all, q64_cache["train"]["q_b"]], dim=1)
    q32_hi_all = (
        F.interpolate(stiefel["train"]["q32"].float(), size=(32, 32), mode="bilinear", align_corners=False).half()
        if mode == "upsample16" else q64_cache["train"]["q32_hi"]
    )
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

    pc_sv = torch.load(V32_PCORE_CKPT, map_location="cpu", weights_only=False)
    pcore = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=True, normalize_u=True).to(device)
    pcore.load_state_dict(pc_sv["ema"], strict=True)
    pcore.train().requires_grad_(True)

    c_sv = torch.load(init_p2_ckpt, map_location="cpu", weights_only=False)
    d_q16 = int(c_sv.get("d_q16", 64 if use_q64 else 32))
    ctrl = SemanticControlAdapter(d_q16=d_q16, d_q32=32).to(device)
    ctrl.load_state_dict(c_sv["ctrl_ema"], strict=True)
    ctrl.train().requires_grad_(True)

    model = PCoreWithSemanticControl(pcore, ctrl).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    params_list = list(model.parameters())
    opt = torch.optim.AdamW([
        {"params": list(ctrl.parameters()), "lr": ctrl_lr},
        {"params": list(pcore.parameters()), "lr": backbone_lr},
    ], betas=(0.9, 0.99), weight_decay=1.0e-4)

    micro, accum = tier_micro_accum(batch_size, default_micro_24gb=128, max_micro_12gb=32, max_micro_8gb=16, device=device)
    n_pc = int(round(batch_size * 0.15))
    n_self = batch_size - n_pc
    self_counts = _build_style_counts(n_self)
    t0 = time.monotonic()

    for step in range(1, updates + 1):
        frac = step / float(max(1, updates))
        mul = 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * frac))
        opt.param_groups[0]["lr"] = ctrl_lr * mul
        opt.param_groups[1]["lr"] = backbone_lr * mul

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
        u_in_full = torch.where(torch.rand(batch_size, 1) < 0.75, u_ex_full, sub_raw417[tid_full, sub_k].float())

        perm = torch.randperm(batch_size)
        z1_full, tid_full = z1_full[perm], tid_full[perm]
        q16_full, q32_hi_full = q16_full[perm], q32_hi_full[perm]
        y16_full, u_in_full = y16_full[perm], u_in_full[perm]

        opt.zero_grad(set_to_none=True)
        ph_acc, st_acc = 0.0, 0.0
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
                pred_v = model(zt, y16_aug, q16_aug, tid, u_in, tau, q32_hi=q32_hi_aug, drop_style=ds, drop_content=dc, layout_scale=lscale, gamma=gamma_b)
                phuber = _pseudo_huber_velocity(pred_v, target_v, delta=0.05)
                z1_hat = (zt + (1.0 - tau).view(mb, 1, 1, 1) * pred_v.float()).clamp(-5.5, 5.5)
                y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                w_st = (tau > 0.25).float() * (~dc).float() * y16_clean_m.float() * (gamma_b > 0.75).float()
                st_loss = ((y16_hat - y16).abs().mean(dim=(1, 2, 3)) * w_st).sum() / w_st.sum().clamp_min(1.0)
                loss_main = (phuber + 0.10 * st_loss) / float(accum)
            loss_main.backward()
            ph_acc += float(phuber.item()) / float(accum)
            st_acc += float(st_loss.item()) / float(accum)

        gn = torch.nn.utils.clip_grad_norm_(params_list, 1.0, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema_model.parameters()), [p.detach() for p in params_list], 0.004)
        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[{arm_name}] {step:4d}/{updates} phuber={ph_acc:.4f} st={st_acc:.4f} gn={float(gn):.2f}", flush=True)
        if step % 500 == 0 and step < updates:
            atomic_save({"pcore_ema": ema_model.pcore.state_dict(), "ctrl_ema": ema_model.ctrl.state_dict(), "step": step, "mode": mode, "d_q16": d_q16}, run_dir / f"{arm_name}_s{step}.pt")

    assert_vram_budget(device=device, tag=arm_name)
    atomic_save({"pcore_ema": ema_model.pcore.state_dict(), "ctrl_ema": ema_model.ctrl.state_dict(), "step": updates, "mode": mode, "d_q16": d_q16}, ckpt_p)
    del model, ema_model, pcore, ctrl, opt, swd_critic, pert_cache, pert_vars
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt_p


@torch.no_grad()
def run_full_p3_audit(
    arm_tag: str,
    ckpt_p: Path,
    *,
    is_joint_ckpt: bool,
    mode: str,
    route_mode: str = "argmax",
    use_q64: bool = False,
    steps: int = 8,
    device: str = "cuda",
) -> dict:
    """Full V33 audit: multi-tau0 (incl. C~0.70/0.61/0.50), Wrong-Target, Correct/Zero/Shuffled causal probe, and Cold-Start Speed."""
    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    re_assets = build_or_load_routing_and_energy(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]

    sv = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    d_q16 = int(sv.get("d_q16", 64 if use_q64 else 32))
    pcore = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=True, normalize_u=True).to(device)
    if is_joint_ckpt:
        pcore.load_state_dict(sv["pcore_ema"], strict=True)
    else:
        pcore.load_state_dict(torch.load(V32_PCORE_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    ctrl = SemanticControlAdapter(d_q16=d_q16, d_q32=32).to(device)
    ctrl.load_state_dict(sv["ctrl_ema"], strict=True)
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
    q16_in = torch.cat([q16_750, q64_cache["test750"]["q_b"].float()], dim=1) if use_q64 else q16_750
    cnx_gray_750 = stiefel["test750"]["cnx_gray"].float()
    q32_hi_750 = F.interpolate(q16_750, size=(32, 32), mode="bilinear", align_corners=False) if mode == "upsample16" else q64_cache["test750"]["q32_hi"].float()
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    draft_cache: dict[tuple[float, bool], torch.Tensor] = {}

    # 1. Multi-anchor frontier + wrong-target + causal probes (correct / zero / shuffled)
    specs = [
        ("tau084", 0.84, False, "correct"),
        ("tau079", 0.79, False, "correct"),
        ("tau075", 0.75, False, "correct"),
        ("tau065", 0.65, False, "correct"),
        ("tau050", 0.50, False, "correct"),
        ("tau035", 0.35, False, "correct"),
        ("tau000", 0.00, False, "correct"),
        ("tau079_wrong", 0.79, True, "correct"),
        ("tau079_zero", 0.79, False, "zero"),
        ("tau079_shuf", 0.79, False, "shuffled"),
        ("tau000_zero", 0.00, False, "zero"),
        ("tau000_shuf", 0.00, False, "shuffled"),
    ]
    shuf_perm = torch.randperm(len(tasks), generator=torch.Generator().manual_seed(SEED_V33 + 88))
    results = {}
    for lbl, t0_v, wrong, cond_mode in specs:
        full_tag = f"{arm_tag}_{lbl}"
        m_dir = OUT / "measured" / full_tag
        if not (m_dir / "summary_v33.json").exists():
            if abs(t0_v - 0.84) < 0.01:
                _, kw = p2_schedule_for_tau0(0.79)
                kw["tau0"] = 0.84
                kw["draft_lambda"] = 0.80
                kw["wc"] = 3.0
                kw["ws"] = 3.5
            else:
                _, kw = p2_schedule_for_tau0(t0_v)
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
                u_st, _, _ = match_cnx_k8_csls(cnx_gray_750[sl].to(device).float(), ti, cnx_bank, re_assets["csls_r"], mode=route_mode)
                ze = ze_pool[sl].to(device).float() if t0_v > 0.0 else None
                if cond_mode == "correct":
                    q16_b = q16_in[sl].to(device).float()
                    q32_b = q32_hi_750[sl].to(device).float()
                elif cond_mode == "zero":
                    q16_b = torch.zeros_like(q16_in[sl], device=device, dtype=torch.float32)
                    q32_b = torch.zeros_like(q32_hi_750[sl], device=device, dtype=torch.float32)
                else:
                    idx_s = shuf_perm[sl]
                    q16_b = q16_in[idx_s].to(device).float()
                    q32_b = q32_hi_750[idx_s].to(device).float()
                zo = integrate_v33_trajectory(
                    model, ze, eps_750[sl].to(device).float(),
                    y16_750[sl].to(device).float(), q16_b,
                    ti, u_st, steps=steps, solver="dpm2m", q32_hi=q32_b, **kw_call,
                )
                writer.write_batch(zo, tasks[sl])
            writer.finalize(arm=full_tag, tau0=t0_v, steps=steps, wrong=wrong, cond_mode=cond_mode)
        results[lbl] = measure_and_summarize(full_tag, device=device)

    # 2. Cold-start RGB->RGB throughput (batch=1 & batch=16, full CFG compute)
    vae = load_a0_encoder(device)
    cnx_ext = ConvNeXtMultiScaleExtractor(device=device).eval()
    w32_st = stiefel["W_stiefel_cnx"].to(device).float()
    whi32_st = q64_cache["W_hi32"].to(device).float()
    w_luma = v18["luma_proj"]["w"].to(device).float()
    b_luma = v18["luma_proj"]["b"].to(device).float()
    _, kw79 = p2_schedule_for_tau0(0.79)
    kw_call79 = {k: v for k, v in kw79.items() if k != "draft_lambda"}
    lat = {}
    for bs in (1, 16):
        rgb_in = torch.rand(bs, 3, 512, 512, device=device)
        tid = torch.full((bs,), 2, dtype=torch.long, device=device)
        eps = torch.randn(bs, 4, 64, 64, device=device)
        lam_b = torch.full((bs,), float(kw79["draft_lambda"]), device=device)
        times = []
        for it in range(5 + 15):
            torch.cuda.synchronize(device); ts0 = time.perf_counter()
            z_b = (vae.encode((rgb_in.half() * 2.0 - 1.0)).latent_dist.mean * 0.18215).float()
            c64_b = rgb_to_structure_64(rgb_in)
            y16_b = project_latent_y16_lcn(z_b, w_luma, b_luma)
            g16_b = cnx_ext.extract_gray_layout(rgb_in)
            g32_b = cnx_ext.extract_gray_layout_32(rgb_in)
            q16_b = extract_cnx_q32(g16_b, w32_st)
            q32_b = extract_cnx_q32_hi(g32_b, whi32_st)
            u_b, _, _ = match_cnx_k8_csls(g16_b, tid, cnx_bank, re_assets["csls_r"], mode=route_mode)
            ze_b = ecore(z_b, c64_b, q16_b, tid, lam_b, raw_stats, priors, noise=eps)
            zo_b = integrate_v33_trajectory(model, ze_b, eps, y16_b, q16_b, tid, u_b, steps=steps, solver="dpm2m", q32_hi=q32_b, **kw_call79)
            _ = decode_s10(dec, zo_b, batch=bs, output_cpu=False)
            torch.cuda.synchronize(device)
            if it >= 5:
                times.append((time.perf_counter() - ts0) * 1000.0 / bs)
        mean_ms = sum(times) / len(times)
        lat[f"b{bs}_ms"] = round(mean_ms, 2)
        lat[f"b{bs}_ips"] = round(1000.0 / mean_ms, 2)

    del model, pcore, ctrl, ecore, dec, vae, cnx_ext
    gc.collect()
    torch.cuda.empty_cache()

    ci_v32 = paired_ci_vs_baseline(OUT / f"measured/{arm_tag}_tau079/eval.json", V32_OUT / "measured/PCore_NoDINO_s085_c10/eval.json")
    sg_eval = ROOT / "docs/baseline/StyleGallery/eval_stylegallery.json"
    ci_sg = paired_ci_vs_baseline(OUT / f"measured/{arm_tag}_tau079/eval.json", sg_eval) if sg_eval.exists() else {}

    return {"levels": results, "cold_start_latency_4090": lat, "ci_vs_v32_s085_c10": ci_v32, "ci_vs_stylegallery": ci_sg}


def run_p3(*, force: bool = False, device: str = "cuda") -> dict:
    sanitize_cuda_env()
    t0 = time.monotonic()

    p0_p, p1_p, p2_p = OUT / "p0_report.json", OUT / "p1_report.json", OUT / "p2_report.json"
    if not p2_p.exists():
        raise FileNotFoundError(f"Missing P2 report: {p2_p}")
    p2_rep = read_json(p2_p)
    p0_rep = read_json(p0_p) if p0_p.exists() else {}
    p1_rep = read_json(p1_p) if p1_p.exists() else {}

    push_check = check_p2_frontier_push(p2_rep)
    print("[P3 Gate Check]", push_check, flush=True)
    if not push_check["pushed_frontier"] and not force:
        out = {
            "updated_utc": utc(),
            "status": "SKIPPED_NO_FRONTIER_PUSH",
            "frontier_check": push_check,
        }
        save_json(OUT / "p3_report.json", out)
        return out

    # Select winning configuration from P0/P1/P2
    route_mode = "csls" if p0_rep.get("p0_1_routing", {}).get("s085_net_positive", False) else "argmax"
    use_q64 = bool(p1_rep.get("gate", {}).get("Q1_q64_plus", {}).get("pass_gate", False))
    w_name = p2_rep["winner_arm"]
    w_info = p2_rep["arms"][w_name]
    w_mode = w_info["mode"]
    w_pert = float(w_info["style_perturb_prob"])

    # 1. Extend winning mechanism to 7000 steps (Seed 0)
    rep_7k_s0 = train_p2_arm("P3_Winner_7k_s0", w_mode, style_perturb_prob=w_pert, use_q64=use_q64, updates=7000, seed_offset=0, device=device)
    # 2. Joint fine-tune from 3500-step winner if frozen backbone bottlenecks
    joint_ckpt = _train_joint_finetune("P3_Joint_7k_s0", Path(w_info["ckpt"]), w_mode, w_pert, use_q64=use_q64, updates=3500, seed_offset=1, device=device)
    # 3. Second seed verification (7000 steps, Seed 1)
    rep_7k_s1 = train_p2_arm("P3_Winner_7k_s1", w_mode, style_perturb_prob=w_pert, use_q64=use_q64, updates=7000, seed_offset=777, device=device)

    audit_7k_s0 = run_full_p3_audit("P3_Audit_7k_s0", Path(rep_7k_s0["ckpt"]), is_joint_ckpt=False, mode=w_mode, route_mode=route_mode, use_q64=use_q64, steps=8, device=device)
    audit_joint = run_full_p3_audit("P3_Audit_Joint_s0", joint_ckpt, is_joint_ckpt=True, mode=w_mode, route_mode=route_mode, use_q64=use_q64, steps=8, device=device)
    audit_7k_s1 = run_full_p3_audit("P3_Audit_7k_s1", Path(rep_7k_s1["ckpt"]), is_joint_ckpt=False, mode=w_mode, route_mode=route_mode, use_q64=use_q64, steps=8, device=device)

    out = {
        "updated_utc": utc(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "status": "COMPLETED",
        "frontier_check": push_check,
        "selected_config": {"route_mode": route_mode, "use_q64": use_q64, "p2_winner": w_name, "mode": w_mode, "style_perturb_prob": w_pert},
        "audit_7k_seed0": audit_7k_s0,
        "audit_joint_finetune": audit_joint,
        "audit_7k_seed1": audit_7k_s1,
    }
    save_json(OUT / "p3_report.json", out)
    print(f"[P3 DONE] in {out['elapsed_s']:.1f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="CLEAVE V33 P3 Joint Verification & Full Audit")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_p3(force=args.force, device=args.device)


if __name__ == "__main__":
    main()
