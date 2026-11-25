"""CLEAVE V33 P1 (~3h budget): Upgrading Q32 -> Q64+ Orthogonal Complement & Partial Whitening Adapter.

Three arms (frozen V32 P-Core backbone + ~2.3M zero-init 16x16 bottleneck Adapter, 3500 steps):
  - Q0: Q32 + zero_pad(32)           (parameter-count control)
  - Q1: Q32 + Q_b(32) [Q64+]         (orthogonal complement capacity test)
  - Q2: Q32 + partial_whiten_res(32) (alpha=0.5, RMS-calibrated, tests whitening salience loss)
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
from exp.v18.structure import _train_linear_mlp_probe
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
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
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
from exp.v33_semantic_control.assets import build_or_load_q64_hi32_cache
from exp.v33_semantic_control.models import PCoreWithQ64Adapter, Q64ResidualAdapter, integrate_v33_trajectory
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc


def _gram_feat_64(t4d: torch.Tensor) -> torch.Tensor:
    fl = t4d.float().flatten(2)
    G = torch.bmm(fl, fl.transpose(1, 2)) / 256.0
    ii = torch.triu_indices(fl.shape[1], fl.shape[1])
    return G[:, ii[0], ii[1]]


def _spatial_relation_probe(q_tr: torch.Tensor, g_tr: torch.Tensor, q_ho: torch.Tensor, g_ho: torch.Tensor) -> float:
    """Measure holdout spatial-affinity R^2 between flattened spatial cosine matrices of Q vs raw ConvNeXt."""
    def _pair_sim(x: torch.Tensor) -> torch.Tensor:
        # 4x4 pooled spatial tokens -> 16 tokens -> 120 upper-tri pairwise similarities
        p = F.normalize(F.avg_pool2d(x.float(), 4).flatten(2).transpose(1, 2), dim=-1)
        sim = torch.bmm(p, p.transpose(1, 2))
        ii = torch.triu_indices(16, 16, offset=1)
        return sim[:, ii[0], ii[1]]

    _ = _pair_sim(q_tr[:32])  # sanity
    sq = _pair_sim(q_ho)
    sg = _pair_sim(g_ho)
    return round(float(F.cosine_similarity(sq, sg, dim=-1).mean().item()), 4)


def run_p1_representation_probes(assets: dict, q64_cache: dict, v18: dict) -> dict:
    """Gram Probe (5-way style leakage) + Spatial Relation Probe on train_4400 -> holdout_300."""
    tr_sid = v18["train"]["sid"].long()
    ho_sid = v18["holdout"]["sid"].long()
    q32_tr = assets["stiefel"]["train"]["q32"].float()
    q32_ho = assets["stiefel"]["holdout"]["q32"].float()
    qb_tr = q64_cache["train"]["q_b"].float()
    qb_ho = q64_cache["holdout"]["q_b"].float()
    pw_tr = q64_cache["train"]["pw_res"].float()
    pw_ho = q64_cache["holdout"]["pw_res"].float()
    g_tr = assets["stiefel"]["train"]["cnx_gray"].float()
    g_ho = q64_cache["holdout"]["cnx_gray"].float()

    q0_tr, q0_ho = torch.cat([q32_tr, torch.zeros_like(q32_tr)], 1), torch.cat([q32_ho, torch.zeros_like(q32_ho)], 1)
    q1_tr, q1_ho = torch.cat([q32_tr, qb_tr], 1), torch.cat([q32_ho, qb_ho], 1)
    q2_tr, q2_ho = torch.cat([q32_tr, pw_tr], 1), torch.cat([q32_ho, pw_ho], 1)

    with torch.enable_grad():
        leak_q0 = _train_linear_mlp_probe(_gram_feat_64(q0_tr), tr_sid, _gram_feat_64(q0_ho), ho_sid)
        leak_q1 = _train_linear_mlp_probe(_gram_feat_64(q1_tr), tr_sid, _gram_feat_64(q1_ho), ho_sid)
        leak_q2 = _train_linear_mlp_probe(_gram_feat_64(q2_tr), tr_sid, _gram_feat_64(q2_ho), ho_sid)

    sp_q0 = _spatial_relation_probe(q0_tr, g_tr, q0_ho, g_ho)
    sp_q1 = _spatial_relation_probe(q1_tr, g_tr, q1_ho, g_ho)
    sp_q2 = _spatial_relation_probe(q2_tr, g_tr, q2_ho, g_ho)

    orth_err = float(
        (torch.bmm(q1_ho.flatten(2), q1_ho.flatten(2).transpose(1, 2)) / 256.0 - torch.eye(64)).abs().max().item()
    )
    return {
        "gram_5way_leak_chance_0.20": {
            "Q0_zero_pad": round(leak_q0, 4),
            "Q1_q64_plus": round(leak_q1, 4),
            "Q2_partial_whiten": round(leak_q2, 4),
        },
        "spatial_relation_cos": {
            "Q0_zero_pad": sp_q0,
            "Q1_q64_plus": sp_q1,
            "Q2_partial_whiten": sp_q2,
        },
        "q64_plus_max_gram_err": round(orth_err, 6),
    }


def _load_frozen_v32_pcore(device: str = "cuda") -> PCoreSLFUNet:
    pc_sv = torch.load(V32_PCORE_CKPT, map_location="cpu", weights_only=False)
    pcore = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=True, normalize_u=True).to(device)
    pcore.load_state_dict(pc_sv["ema"], strict=True)
    pcore.eval().requires_grad_(False)
    return pcore


@torch.no_grad()
def evaluate_p1_arm(
    arm_name: str,
    ckpt_p: Path,
    mode: str,
    *,
    device: str = "cuda",
) -> dict:
    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]

    pcore = _load_frozen_v32_pcore(device=device)
    sv = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    adapter = Q64ResidualAdapter(in_q_ch=64, w_mid16=432, cond_dim=320, inner_ch=256).to(device)
    adapter.load_state_dict(sv["adapter_ema"], strict=True)
    adapter.eval().requires_grad_(False)
    model = PCoreWithQ64Adapter(pcore, adapter).to(device).eval().requires_grad_(False)

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
    q32_750 = stiefel["test750"]["q32"].float()
    cnx_gray_750 = stiefel["test750"]["cnx_gray"].float()
    if mode == "zero":
        q_extra_750 = torch.zeros_like(q32_750)
    elif mode == "q64_plus":
        q_extra_750 = q64_cache["test750"]["q_b"].float()
    elif mode == "partial_whiten":
        q_extra_750 = q64_cache["test750"]["pw_res"].float()
    else:
        raise ValueError(f"Unknown P1 mode: {mode!r}")

    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    draft_cache: dict[tuple[float, bool], torch.Tensor] = {}

    eval_specs = [
        ("s085_c10", 0.85, S085_C10_KW, False),
        ("s100", 1.00, default_knob_schedule(1.00, use_gamma=True), False),
        ("s125", 1.25, default_knob_schedule(1.25, use_gamma=True), False),
        ("s085_c10_wrong", 0.85, S085_C10_KW, True),
    ]
    out = {}
    for lvl_tag, sv_val, kw, wrong in eval_specs:
        full_tag = f"{arm_name}_{lvl_tag}"
        m_dir = OUT / "measured" / full_tag
        if not (m_dir / "summary_v33.json").exists():
            lam = round(float(kw["draft_lambda"]), 3)
            if kw["tau0"] > 0.0 and (lam, wrong) not in draft_cache:
                chunks = []
                for st in range(0, len(tasks), 32):
                    sl = slice(st, min(st + 32, len(tasks)))
                    ti = pick_wrong_cross_tid(sid_750[sl], tid_750[sl]).to(device) if wrong else tid_750[sl].to(device)
                    chunks.append(ecore(
                        z_src[sl].to(device).float(), c64_750[sl].to(device).float(),
                        q32_750[sl].to(device).float(), ti,
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
                ze = ze_pool[sl].to(device).float() if kw["tau0"] > 0.0 else None
                zo = integrate_v33_trajectory(
                    model, ze, eps_750[sl].to(device).float(),
                    y16_750[sl].to(device).float(), q32_750[sl].to(device).float(),
                    ti, u_st, steps=6, solver="dpm2m",
                    q_extra=q_extra_750[sl].to(device).float(), **kw_call,
                )
                writer.write_batch(zo, tasks[sl])
            writer.finalize(arm=full_tag, s=sv_val, wrong=wrong, schedule=kw)
        res = measure_and_summarize(full_tag, device=device)
        out[lvl_tag] = res
        print(
            f"[P1 Eval] {full_tag:28s} | C={res['dino_c']:.4f} S={res['dino_s']:.4f} Top5={res['dino_s_top5']:.4f} "
            f"Top1={res['csd_top1']*100:5.2f}% PredSrc={res['pred_eq_src']*100:4.1f}% Same={res['same150_top1']*100:5.1f}%",
            flush=True,
        )

    del model, pcore, adapter, ecore, dec
    gc.collect()
    torch.cuda.empty_cache()
    return out


def train_p1_arm(
    arm_name: str,
    mode: str,
    *,
    updates: int = 3500,
    lr: float = 3.0e-4,
    batch_size: int = 128,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict:
    sanitize_cuda_env()
    seed_all(SEED_V33 + 100)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    run_dir = OUT / "p1" / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_p = run_dir / f"{arm_name}_s{updates}.pt"
    rep_p = run_dir / "arm_report.json"
    if rep_p.exists() and ckpt_p.exists() and smoke_steps == 0:
        print(f"[{arm_name}] Reusing completed P1 arm.", flush=True)
        return read_json(rep_p)

    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    q32_all = stiefel["train"]["q32"]
    if mode == "zero":
        q_extra_all = torch.zeros_like(q32_all)
    elif mode == "q64_plus":
        q_extra_all = q64_cache["train"]["q_b"]
    elif mode == "partial_whiten":
        q_extra_all = q64_cache["train"]["pw_res"]
    else:
        raise ValueError(f"Unknown P1 mode: {mode!r}")

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
    pc_q32 = q32_all[pc_ridx]
    pc_qe = q_extra_all[pc_ridx]

    swd_critic = _HighTEndpointSWDCritic(device=device)
    swd_critic.set_moment_scale(torch.cat([z_all[ci[:10]] for ci in core_cls], dim=0))

    pcore = _load_frozen_v32_pcore(device=device)
    adapter = Q64ResidualAdapter(in_q_ch=64, w_mid16=432, cond_dim=320, inner_ch=256).to(device)
    n_adapter_params = adapter.n_params()
    adapter_ema = copy.deepcopy(adapter).eval().requires_grad_(False)
    model = PCoreWithQ64Adapter(pcore, adapter).to(device)
    params_list = list(adapter.parameters())
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

        z1_full = torch.cat([z_all[idx_self].float(), pc_zreal[p_pick].float()], dim=0)
        tid_full = torch.cat([sid_all[idx_self], pc_tid[p_pick]], dim=0)
        q32_full = torch.cat([q32_all[idx_self].float(), pc_q32[p_pick].float()], dim=0)
        qe_full = torch.cat([q_extra_all[idx_self].float(), pc_qe[p_pick].float()], dim=0)
        y16_full = torch.cat([y16_all[idx_self].float(), pc_y16[p_pick].float()], dim=0)
        u_ex_full = torch.cat([norm417_all[idx_self].float(), norm417_all[pc_ridx[p_pick]].float()], dim=0)

        sub_k = torch.randint(0, 8, (batch_size,))
        u_sub_full = sub_raw417[tid_full, sub_k].float()
        u_in_full = torch.where(torch.rand(batch_size, 1) < 0.75, u_ex_full, u_sub_full)

        perm = torch.randperm(batch_size)
        z1_full, tid_full = z1_full[perm], tid_full[perm]
        q32_full, qe_full = q32_full[perm], qe_full[perm]
        y16_full, u_in_full = y16_full[perm], u_in_full[perm]

        opt.zero_grad(set_to_none=True)
        ph_acc, st_acc, pix_val = 0.0, 0.0, 0.0

        for a_i in range(accum):
            sl = slice(a_i * micro, (a_i + 1) * micro)
            mb = micro
            z1 = z1_full[sl].to(device, non_blocking=True)
            tid = tid_full[sl].to(device, non_blocking=True)
            q32 = q32_full[sl].to(device, non_blocking=True)
            qe = qe_full[sl].to(device, non_blocking=True)
            y16 = y16_full[sl].to(device, non_blocking=True)
            u_in = u_in_full[sl].to(device, non_blocking=True)

            q32_aug = q32 + 0.06 * torch.randn_like(q32)
            q32_aug = q32_aug - q32_aug.mean(dim=(2, 3), keepdim=True)
            qe_aug = (qe + 0.06 * torch.randn_like(qe)) if mode != "zero" else qe
            if mode != "zero":
                qe_aug = qe_aug - qe_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug, y16_clean_m = _blur_or_drop_y16(y16, drop_prob=0.40)

            ds = torch.rand(mb, device=device) < 0.15
            dc = torch.rand(mb, device=device) < 0.30
            lscale = torch.where(dc, torch.zeros(mb, device=device), 0.25 + 0.75 * torch.rand(mb, device=device))
            gamma_b = torch.where(torch.rand(mb, device=device) < 0.40, torch.ones(mb, device=device), 0.15 + 0.85 * torch.rand(mb, device=device))

            z0 = torch.randn_like(z1)
            tau = torch.rand(mb, device=device)
            zt = (1.0 - tau).view(mb, 1, 1, 1) * z0 + tau.view(mb, 1, 1, 1) * z1
            target_v = z1 - z0

            with bf16_autocast(device):
                pred_v = model(
                    zt, y16_aug, q32_aug, tid, u_in, tau,
                    q_extra=qe_aug, drop_style=ds, drop_content=dc, layout_scale=lscale, gamma=gamma_b,
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
                            zt[pick], y16_aug[pick], q32_aug[pick], tid[pick], u_in[pick], tau[pick],
                            q_extra=qe_aug[pick], drop_style=ds[pick], drop_content=dc[pick],
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
            torch._foreach_lerp_(list(adapter_ema.parameters()), [p.detach() for p in params_list], 0.004)

        if step == 1 or step % 400 == 0 or step == target_steps:
            dt = (time.monotonic() - t0) / step
            print(f"[{arm_name}] {step:4d}/{target_steps} phuber={ph_acc:.4f} st={st_acc:.4f} pix={pix_val:.4f} gn={float(gn):.2f} ({dt:.3f}s/st)", flush=True)

        if smoke_steps == 0 and step % 500 == 0 and step < updates:
            atomic_save({"adapter_ema": adapter_ema.state_dict(), "step": step, "mode": mode, "params": n_adapter_params}, run_dir / f"{arm_name}_s{step}.pt")

    vram_gb = assert_vram_budget(device=device, tag=arm_name)
    elapsed_min = round((time.monotonic() - t0) / 60.0, 2)
    del model, pcore, adapter, opt, swd_critic
    gc.collect()
    torch.cuda.empty_cache()

    if smoke_steps > 0:
        del adapter_ema
        torch.cuda.empty_cache()
        return {"arm": arm_name, "mode": mode, "adapter_params": n_adapter_params, "peak_vram_gb": round(vram_gb, 3), "elapsed_min": elapsed_min}

    atomic_save({"adapter_ema": adapter_ema.state_dict(), "step": updates, "mode": mode, "params": n_adapter_params}, ckpt_p)
    del adapter_ema
    gc.collect()
    torch.cuda.empty_cache()

    eval_out = evaluate_p1_arm(arm_name, ckpt_p, mode, device=device)
    rep = {
        "updated_utc": utc(),
        "arm": arm_name,
        "mode": mode,
        "adapter_params": n_adapter_params,
        "train_min": elapsed_min,
        "peak_vram_gb": round(vram_gb, 3),
        "ckpt": str(ckpt_p),
        "levels": eval_out,
    }
    save_json(rep_p, rep)
    return rep


def run_p1(*, smoke: bool = False, device: str = "cuda") -> dict:
    sanitize_cuda_env()
    t0 = time.monotonic()
    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    probes = run_p1_representation_probes(assets, q64_cache, v18)
    print("[P1 Probes]", probes, flush=True)

    arms_spec = [
        ("P1_Q0_zero_pad", "zero"),
        ("P1_Q1_q64_plus", "q64_plus"),
        ("P1_Q2_partial_whiten", "partial_whiten"),
    ]
    if smoke:
        smoke_res = {}
        for name, md in arms_spec[:2]:
            smoke_res[name] = train_p1_arm(name, md, smoke_steps=20, device=device)
        return {"probes": probes, "smoke": smoke_res}

    arm_reports = {}
    for name, md in arms_spec:
        arm_reports[name] = train_p1_arm(name, md, updates=3500, device=device)

    # Gate check vs Q0 and V32 baseline at s085_c10 / s100 / s125
    q0 = arm_reports["P1_Q0_zero_pad"]["levels"]
    q1 = arm_reports["P1_Q1_q64_plus"]["levels"]
    q2 = arm_reports["P1_Q2_partial_whiten"]["levels"]
    v32_ledger = read_json(V32_OUT / "v32_final_ledger.json")["spectrum"]["PCore_NoDINO_s085_c10"]

    def _gate_arm(res_lvls: dict, leak_val: float) -> dict:
        ds_85 = res_lvls["s085_c10"]["dino_s"] - q0["s085_c10"]["dino_s"]
        dc_85 = res_lvls["s085_c10"]["dino_c"] - q0["s085_c10"]["dino_c"]
        ds_100 = res_lvls["s100"]["dino_s"] - q0["s100"]["dino_s"]
        dc_100 = res_lvls["s100"]["dino_c"] - q0["s100"]["dino_c"]
        # Check frontier shift (same C: dS >= 0.01 OR same S: dC >= 0.025) & no style leak spike
        leak_ok = (leak_val <= 0.30) and ((res_lvls["s085_c10"]["pred_eq_src"] - q0["s085_c10"]["pred_eq_src"]) <= 0.03)
        passed = leak_ok and (
            (ds_85 >= 0.01 and dc_85 >= -0.005)
            or (dc_85 >= 0.025 and ds_85 >= -0.005)
            or (ds_100 >= 0.01 and dc_100 >= -0.005)
            or (dc_100 >= 0.025 and ds_100 >= -0.005)
        )
        return {
            "pass_gate": bool(passed),
            "leak_ok": bool(leak_ok),
            "delta_vs_q0_s085": {"dino_c": round(dc_85, 4), "dino_s": round(ds_85, 4)},
            "delta_vs_q0_s100": {"dino_c": round(dc_100, 4), "dino_s": round(ds_100, 4)},
            "delta_vs_v32_s085": {
                "dino_c": round(res_lvls["s085_c10"]["dino_c"] - v32_ledger["dino_c"], 4),
                "dino_s": round(res_lvls["s085_c10"]["dino_s"] - v32_ledger["dino_s"], 4),
            },
        }

    gate = {
        "Q1_q64_plus": _gate_arm(q1, probes["gram_5way_leak_chance_0.20"]["Q1_q64_plus"]),
        "Q2_partial_whiten": _gate_arm(q2, probes["gram_5way_leak_chance_0.20"]["Q2_partial_whiten"]),
        "ci_q1_vs_q0_s085": paired_ci_vs_baseline(OUT / "measured/P1_Q1_q64_plus_s085_c10/eval.json", OUT / "measured/P1_Q0_zero_pad_s085_c10/eval.json"),
        "ci_q2_vs_q0_s085": paired_ci_vs_baseline(OUT / "measured/P1_Q2_partial_whiten_s085_c10/eval.json", OUT / "measured/P1_Q0_zero_pad_s085_c10/eval.json"),
    }
    report = {
        "updated_utc": utc(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "probes": probes,
        "arms": arm_reports,
        "gate": gate,
    }
    save_json(OUT / "p1_report.json", report)
    print(f"[P1 DONE] in {report['elapsed_s']:.1f}s | Gate Q1={gate['Q1_q64_plus']['pass_gate']} Q2={gate['Q2_partial_whiten']['pass_gate']}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="CLEAVE V33 P1 Q64+ Adapter")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_p1(smoke=args.smoke, device=args.device)


if __name__ == "__main__":
    main()
