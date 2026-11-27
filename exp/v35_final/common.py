"""Shared evaluation, content-representation, Holdout-750, E2E profiler, and 44M training infra for CLEAVE V35.

Zero-DINO AST hygiene maintained strictly across all helpers.
"""
from __future__ import annotations

import copy
import gc
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import coarse_y16_lcn, project_latent_y16_lcn
from exp.v25.r14_e2e_benchmark import load_a0_encoder
from exp.v27_single.bare_q32 import load_cache
from exp.v28_unified.student_models import build_student_model
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v29_biglittle.phase1_train import (
    _HighTEndpointSWDCritic,
    _blur_or_drop_y16,
    _build_style_counts,
)
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor, extract_cnx_q32, match_cnx_k8_submode
from exp.v32_nodino.diag_q32_6arm import perturb_q32
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v34_frontier.train_common import run_pcore_train_step, schedule_for_v34
from exp.v35_final import (
    OUT,
    POOL_C_PATH,
    SEED_V35,
    V32_ECORE_CKPT,
    RawWriterV35,
    assert_vram_budget,
    atomic_save,
    build_or_load_analytic_priors,
    build_pcore_startpoint,
    measure_and_summarize_batch_v35,
)
from src.eval.calib import build_or_load_calib_spec, run_calib_measure_batch_subprocess
from src.eval.holdout import build_or_load_holdout_spec, run_holdout_measure_subprocess
from src.utils import auto_infer_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import auto_vram_budget_gb, read_json, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, vram_total_gb
from src.utils.structure import local_contrast_norm, rgb_to_gray, rgb_to_structure_64


def load_pcore_v35(ckpt_path: Path, *, weight_key: str = "ema", device: str = "cuda") -> tuple[PCoreSLFUNet, dict[str, Any]]:
    """Load 22M / 44M PCoreSLFUNet checkpoint cleanly (auto-detects V32 vs V34/V35 formats)."""
    sv = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sv.get("cfg") or {"base_dim": 144, "u_dim": 417, "cond_dim": 320}
    pcore = PCoreSLFUNet(
        base_dim=int(cfg["base_dim"]),
        u_dim=int(cfg["u_dim"]),
        cond_dim=int(cfg["cond_dim"]),
        use_gamma=True,
        normalize_u=True,
    ).to(device)
    state = sv.get(weight_key) or sv["ema"]
    pcore.load_state_dict(state, strict=True)
    pcore.eval().requires_grad_(False)
    return pcore, cfg


def make_v35_schedule(
    knob_s: float,
    *,
    ws_mul: float = 1.0,
    wc_mul: float = 1.0,
    dtau0: float = 0.0,
) -> dict[str, Any]:
    """Single-knob schedule mapping s -> (tau0, ws, wc, wsp, rescale_phi, gamma, lscale, draft_lambda).

    Guarantees s=0 exact identity, keeps s>=0.75 and s=0.65 bit-identical to V34, and smoothly
    interpolates sub-0.75 (tau0, draft_lambda) for high-content & E/P handoff evaluations.
    """
    sv = float(knob_s)
    if sv <= 0.0:
        return {"tau0": 1.0, "ws": 1.0, "wc": 1.0, "wsp": 0.0, "rescale_phi": 0.0, "gamma": 1.0, "lscale": 1.0, "draft_lambda": 0.0}
    kw = dict(schedule_for_v34(sv))
    if sv < 0.75 - 1e-6:
        sub_knots = ((0.50, 0.920, 0.450), (0.60, 0.895, 0.500), (0.65, 0.870, 0.550), (0.75, 0.835, 0.650))
        sc = max(0.50, sv)
        t0_i, lam_i = sub_knots[-1][1], sub_knots[-1][2]
        for (s0, a0, l0), (s1, a1, l1) in zip(sub_knots, sub_knots[1:]):
            if sc <= s1 + 1e-9:
                u = (sc - s0) / max(s1 - s0, 1e-6)
                t0_i = (1.0 - u) * a0 + u * a1
                lam_i = (1.0 - u) * l0 + u * l1
                break
        kw["tau0"] = round(t0_i, 4)
        kw["draft_lambda"] = round(lam_i, 3)
    kw["ws"] = round(float(kw["ws"]) * float(ws_mul), 4)
    kw["wc"] = round(float(kw["wc"]) * float(wc_mul), 4)
    if float(kw["tau0"]) > 0.0:
        kw["tau0"] = round(max(0.05, min(0.95, float(kw["tau0"]) + float(dtau0))), 4)
    return kw


def load_v32_ecore_and_priors(device: str = "cuda") -> tuple[torch.nn.Module, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device).float()
    priors_ec = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device).eval().requires_grad_(False)
    ecore.load_state_dict(ec_sv["ema"], strict=True)
    priors_a = build_or_load_analytic_priors(priors_ec)
    return ecore, raw_stats, priors_ec, priors_a


def compute_whiten32(p16_gray: torch.Tensor, w_stiefel: torch.Tensor) -> torch.Tensor:
    """Plain per-channel spatial whitening on W_stiefel^T @ x without polar Gram orthogonalization.

    Shares identical 32-D linear projection basis as Stiefel Q32, isolating only polar SVD orthogonalization.
    """
    fl = torch.einsum("dk,bdp->bkp", w_stiefel.to(device=p16_gray.device, dtype=torch.float32), p16_gray.float().flatten(2))
    mu = fl.mean(dim=-1, keepdim=True)
    sd = fl.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    q = (fl - mu) / sd
    return q.reshape(p16_gray.shape[0], w_stiefel.shape[1], 16, 16)


def compute_pca32_raw(p16_gray: torch.Tensor, w_pca: torch.Tensor, rms_scale: float) -> torch.Tensor:
    """Raw un-whitened 32-D PCA projection normalized only by global train RMS scalar."""
    fl = torch.einsum("dk,bdp->bkp", w_pca.to(device=p16_gray.device, dtype=torch.float32), p16_gray.float().flatten(2))
    fl = fl - fl.mean(dim=-1, keepdim=True)
    return (fl / max(float(rms_scale), 1e-4)).reshape(p16_gray.shape[0], w_pca.shape[1], 16, 16)


@torch.no_grad()
def build_or_load_content_ablation_reps(device: str = "cuda") -> dict[str, Any]:
    """Build or load matched 32-ch representations (stiefel, whiten32, pca32) across train/holdout/test750."""
    cache_p = OUT / "cache/content_ablation_reps.pt"
    if cache_p.exists():
        return torch.load(cache_p, map_location="cpu", weights_only=False)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    assets = build_or_load_cnx_assets(device=device)
    st = assets["stiefel"]
    w_stiefel = st["W_stiefel_cnx"].float()

    # Build holdout cnx_gray cleanly (zero overlap with train/test750; cached once via ConvNeXt Stage-3)
    if "cnx_gray" in st["holdout"]:
        ho_gray = st["holdout"]["cnx_gray"].half()
    else:
        v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
        z_ho = v18["holdout"]["z"]
        dec = load_s10_decoder(device)
        cnx_ext = ConvNeXtTinyStage3Extractor(device=device).eval()
        chunks = []
        bs = auto_infer_batch(device)
        for s in range(0, len(z_ho), bs):
            sl = slice(s, min(s + bs, len(z_ho)))
            rgb = decode_s10(dec, z_ho[sl].to(device).float(), batch=min(64, sl.stop - sl.start), output_cpu=False).clamp(0.0, 1.0)
            chunks.append(cnx_ext.extract_gray_layout(rgb).cpu().half())
        ho_gray = torch.cat(chunks, 0)
        del dec, cnx_ext
        torch.cuda.empty_cache()

    tr_gray = st["train"]["cnx_gray"].half()
    ts_gray = st["test750"]["cnx_gray"].half()

    # Global RMS of centered linear projection for Raw PCA-32 scale calibration
    fl_tr = torch.einsum("dk,bdp->bkp", w_stiefel.to(device), tr_gray[:512].to(device).float().flatten(2))
    fl_tr = fl_tr - fl_tr.mean(dim=-1, keepdim=True)
    pca_rms = float(fl_tr.square().mean().sqrt().item())

    def _pack(g: torch.Tensor, q_stief: torch.Tensor) -> dict[str, torch.Tensor]:
        wh_l, pc_l = [], []
        for i in range(0, len(g), 256):
            gb = g[i:i + 256].to(device)
            wh_l.append(compute_whiten32(gb, w_stiefel).cpu().half())
            pc_l.append(compute_pca32_raw(gb, w_stiefel, pca_rms).cpu().half())
        return {
            "stiefel": q_stief.cpu().half(),
            "whiten32": torch.cat(wh_l, 0),
            "pca32": torch.cat(pc_l, 0),
        }

    payload = {
        "pca_rms": pca_rms,
        "train": _pack(tr_gray, st["train"]["q32"]),
        "holdout": _pack(ho_gray, st["holdout"]["q32"]),
        "test750": _pack(ts_gray, st["test750"]["q32"]),
    }
    torch.save(payload, cache_p)
    return payload


@torch.no_grad()
def eval_pcore_test750(
    arm_name: str,
    pcore: PCoreSLFUNet,
    *,
    knob_list: tuple[float, ...] = (0.75, 0.85, 1.00, 1.25, 1.50),
    include_tau0: bool = False,
    eval_wrong: bool = True,
    alpha_y: float = 1.0,
    alpha_c: float = 1.0,
    ws_mul: float = 1.0,
    wc_mul: float = 1.0,
    dtau0: float = 0.0,
    steps: int = 6,
    startpoint_mode: str = "ecore_a3",  # "ecore_a3" | "raw_src" | "raw_src_chroma" | "raw_src_a3" | "ecore" | "pure_noise" | "self_ep"
    wrong_draft: bool = False,
    wrong_pcore: bool = False,
    q_mode: str = "correct",
    q32_override: torch.Tensor | None = None,
    shared_drafts: dict[tuple[float, bool], torch.Tensor] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Unified parameterized test750 evaluator covering P0, P1, P3, P4, P5, P6, P7, and P8."""
    sanitize_cuda_env()
    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    q16_base = (q32_override if q32_override is not None else stiefel["test750"]["q32"]).float()
    cnx_gray_750 = stiefel["test750"]["cnx_gray"].float()

    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    dec = load_s10_decoder(device)

    tasks, z_src, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_wrong = pick_wrong_cross_tid(sid_750, tid_750)
    y16_750 = project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    c64_750 = ts["c_64"].float()
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    eval_specs: list[tuple[str, dict[str, Any], bool, bool]] = []
    for kv in knob_list:
        kw = make_v35_schedule(kv, ws_mul=ws_mul, wc_mul=wc_mul, dtau0=dtau0)
        eval_specs.append((f"s{int(round(kv * 100)):03d}", kw, wrong_draft, wrong_pcore))
    if include_tau0:
        eval_specs.append(("tau0", {"tau0": 0.0, "ws": round(4.5 * ws_mul, 4), "wc": round(1.8 * wc_mul, 4), "wsp": 1.35, "rescale_phi": 0.40, "gamma": 0.35, "lscale": 0.72, "draft_lambda": 0.75}, False, False))
    if eval_wrong:
        eval_specs.append(("s085_wrong", make_v35_schedule(0.85, ws_mul=ws_mul, wc_mul=wc_mul, dtau0=dtau0), True, True))

    draft_cache = shared_drafts if shared_drafts is not None else {}
    infer_bs = auto_infer_batch(device)
    pending_meas: list[tuple[str, str]] = []

    for lvl_tag, kw, wd_flag, wp_flag in eval_specs:
        full_tag = f"{arm_name}_{lvl_tag}"
        m_dir = OUT / "measured" / full_tag
        if not (m_dir / "summary_v34.json").exists():
            t0_v = float(kw["tau0"])
            lam = round(float(kw.get("draft_lambda", 0.75)), 3)
            need_ecore = (t0_v > 0.0) and (startpoint_mode in ("ecore_a3", "ecore"))
            need_self_draft = (t0_v > 0.0) and (startpoint_mode == "self_ep")
            if need_ecore and (lam, wd_flag) not in draft_cache:
                chunks = []
                q16_ec = stiefel["test750"]["q32"].float()
                for st in range(0, len(tasks), infer_bs):
                    sl = slice(st, min(st + infer_bs, len(tasks)))
                    ti_d = tid_wrong[sl].to(device) if wd_flag else tid_750[sl].to(device)
                    ze = ecore(
                        z_src[sl].to(device).float(), c64_750[sl].to(device).float(),
                        q16_ec[sl].to(device).float(), ti_d,
                        torch.full((sl.stop - sl.start,), lam, device=device),
                        raw_stats, priors_ec, noise=eps_750[sl].to(device).float(),
                    )
                    chunks.append(ze.cpu().half())
                draft_cache[(lam, wd_flag)] = torch.cat(chunks, 0)
            if need_self_draft and ("self_x1", wd_flag) not in draft_cache:
                # 1-NFE flow jump from tau=0 using CFG on the same P-Core backbone (iso-param shared test)
                chunks_x1 = []
                for st in range(0, len(tasks), infer_bs):
                    sl = slice(st, min(st + infer_bs, len(tasks)))
                    ti_d = (tid_wrong[sl] if wd_flag else tid_750[sl]).to(device)
                    u_d = match_cnx_k8_submode(cnx_gray_750[sl].to(device).float(), ti_d, cnx_bank)
                    ep_d = eps_750[sl].to(device).float()
                    # Integrate 1 step from tau=0 -> 1 with calibrated CFG (ws=3.5, wc=2.4) for 1-NFE draft
                    z_hat = integrate_v33_trajectory(
                        pcore, None, ep_d, y16_750[sl].to(device).float(), q16_base[sl].to(device).float(),
                        ti_d, u_d, tau0=0.0, ws=3.5, wc=2.4, wsp=1.5, rescale_phi=0.35, gamma=1.0, lscale=0.80, steps=1, solver="heun",
                    )
                    chunks_x1.append(z_hat.cpu().half())
                draft_cache[("self_x1", wd_flag)] = torch.cat(chunks_x1, 0)
            ze_pool = draft_cache.get((lam, wd_flag)) if need_ecore else None
            self_x1_pool = draft_cache.get(("self_x1", wd_flag)) if need_self_draft else None

            writer = RawWriterV35(full_tag, dec=dec, device=device)
            kw_call = {k: v for k, v in kw.items() if k != "draft_lambda"}
            for st in range(0, len(tasks), infer_bs):
                sl = slice(st, min(st + infer_bs, len(tasks)))
                ti_p = (tid_wrong[sl] if wp_flag else tid_750[sl]).to(device)
                ti_d = (tid_wrong[sl] if wd_flag else tid_750[sl]).to(device)
                u_st = match_cnx_k8_submode(cnx_gray_750[sl].to(device).float(), ti_p, cnx_bank)
                q_in = perturb_q32(q16_base[sl].to(device).float(), q_mode, seed=SEED_V35 + st)

                if t0_v <= 0.0 or startpoint_mode == "pure_noise":
                    ze_start = None
                    kw_call_eff = {**kw_call, "tau0": 0.0} if startpoint_mode == "pure_noise" else kw_call
                elif startpoint_mode == "raw_src":
                    ze_start = z_src[sl].to(device).float()
                    kw_call_eff = kw_call
                elif startpoint_mode == "raw_src_chroma":
                    ze_start = build_pcore_startpoint(z_src[sl].to(device).float(), ti_d, priors_a, alpha_y=0.0, alpha_c=1.0)
                    kw_call_eff = kw_call
                elif startpoint_mode == "raw_src_a3":
                    ze_start = build_pcore_startpoint(z_src[sl].to(device).float(), ti_d, priors_a, alpha_y=alpha_y, alpha_c=alpha_c)
                    kw_call_eff = kw_call
                elif startpoint_mode == "self_ep":
                    assert self_x1_pool is not None
                    ze_raw = (1.0 - lam) * z_src[sl].to(device).float() + lam * self_x1_pool[sl].to(device).float()
                    ze_start = build_pcore_startpoint(ze_raw, ti_d, priors_a, alpha_y=alpha_y, alpha_c=alpha_c)
                    kw_call_eff = kw_call
                elif startpoint_mode == "ecore":
                    assert ze_pool is not None
                    ze_start = ze_pool[sl].to(device).float()
                    kw_call_eff = kw_call
                else:
                    assert ze_pool is not None
                    ze_start = build_pcore_startpoint(ze_pool[sl].to(device).float(), ti_d, priors_a, alpha_y=alpha_y, alpha_c=alpha_c)
                    kw_call_eff = kw_call

                zo = integrate_v33_trajectory(
                    pcore, ze_start, eps_750[sl].to(device).float(),
                    y16_750[sl].to(device).float(), q_in,
                    ti_p, u_st, steps=steps, solver="dpm2m", **kw_call_eff,
                )
                writer.write_batch(zo, tasks[sl])
            writer.finalize(
                arm=full_tag, wrong_draft=wd_flag, wrong_pcore=wp_flag,
                startpoint_mode=startpoint_mode, q_mode=q_mode,
                schedule=kw, alpha_y=alpha_y, alpha_c=alpha_c, steps=steps,
            )
        pending_meas.append((lvl_tag, full_tag))

    del ecore, dec
    gc.collect()
    torch.cuda.empty_cache()

    batch_res = measure_and_summarize_batch_v35([ft for _, ft in pending_meas], device=device)
    levels: dict[str, Any] = {}
    for lvl_tag, full_tag in pending_meas:
        res = batch_res[full_tag]
        levels[lvl_tag] = res
        print(
            f"[V35 Eval] {full_tag:34s} | C={res['dino_c']:.4f} S={res['dino_s']:.4f} "
            f"Top1={res['csd_top1']*100:5.2f}% LP={res['lpips']:.4f} Same={res['same150_top1']*100:5.1f}%",
            flush=True,
        )
    return levels


@torch.no_grad()
def eval_pcore_train_calib(
    arm_name: str,
    pcore: PCoreSLFUNet,
    *,
    knob_list: tuple[float, ...] = (0.75, 0.85, 1.00, 1.25),
    alpha_y: float = 1.0,
    alpha_c: float = 1.0,
    ws_mul: float = 1.0,
    wc_mul: float = 1.0,
    dtau0: float = 0.0,
    steps: int = 6,
    shared_drafts: dict[float, torch.Tensor] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate strictly on train_calib_300 (60 core train_4400 sources x 5 targets) for P6 calibration."""
    sanitize_cuda_env()
    spec = build_or_load_calib_spec(device=device)
    tasks = spec["tasks"]
    src_idx = spec["src_idx"]

    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr = v18["train"]

    z_src = tr["z"][src_idx].float()
    c64_src = tr["c_64"][src_idx].float()
    q16_src = stiefel["train"]["q32"][src_idx].float()
    cnx_gray_src = stiefel["train"]["cnx_gray"][src_idx].float()
    y16_src = project_latent_y16_lcn(z_src, v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    eps_src = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED + 350))

    src_pos = torch.tensor([t["source_pos"] for t in tasks], dtype=torch.long)
    tid_all = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)

    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    dec = load_s10_decoder(device)
    draft_cache = shared_drafts if shared_drafts is not None else {}
    infer_bs = auto_infer_batch(device)
    calib_items: list[tuple[str, Path, Path, str]] = []

    for kv in knob_list:
        kw = make_v35_schedule(kv, ws_mul=ws_mul, wc_mul=wc_mul, dtau0=dtau0)
        lvl_tag = f"s{int(round(kv * 100)):03d}"
        full_tag = f"{arm_name}_{lvl_tag}"
        raw_dir = OUT / "calib/raw" / full_tag
        m_dir = OUT / "calib/measured" / full_tag
        if not (m_dir / "summary_calib.json").exists():
            t0_v = float(kw["tau0"])
            lam = round(float(kw.get("draft_lambda", 0.75)), 3)
            if t0_v > 0.0 and lam not in draft_cache:
                chunks = []
                for st in range(0, len(tasks), infer_bs):
                    sl = slice(st, min(st + infer_bs, len(tasks)))
                    sp = src_pos[sl]
                    ti = tid_all[sl].to(device)
                    ze = ecore(
                        z_src[sp].to(device), c64_src[sp].to(device), q16_src[sp].to(device),
                        ti, torch.full((len(ti),), lam, device=device), raw_stats, priors_ec,
                        noise=eps_src[sp].to(device),
                    )
                    chunks.append(ze.cpu().half())
                draft_cache[lam] = torch.cat(chunks, 0)
            ze_pool = draft_cache.get(lam)

            writer = RawWriterV35(full_tag, dec=dec, device=device, raw_root=OUT / "calib/raw", expected_tasks=len(tasks))
            kw_call = {k: v for k, v in kw.items() if k != "draft_lambda"}
            for st in range(0, len(tasks), infer_bs):
                sl = slice(st, min(st + infer_bs, len(tasks)))
                sp = src_pos[sl]
                ti = tid_all[sl].to(device)
                u_st = match_cnx_k8_submode(cnx_gray_src[sp].to(device), ti, cnx_bank)
                ze_start = build_pcore_startpoint(ze_pool[sl].to(device).float(), ti, priors_a, alpha_y=alpha_y, alpha_c=alpha_c)
                zo = integrate_v33_trajectory(
                    pcore, ze_start, eps_src[sp].to(device),
                    y16_src[sp].to(device), q16_src[sp].to(device),
                    ti, u_st, steps=steps, solver="dpm2m", **kw_call,
                )
                writer.write_batch(zo, tasks[sl])
            writer.finalize(arm=full_tag, calib=True, schedule=kw, alpha_y=alpha_y, alpha_c=alpha_c, steps=steps)
        calib_items.append((lvl_tag, raw_dir, m_dir, full_tag))

    del ecore, dec
    gc.collect()
    torch.cuda.empty_cache()

    batch_summaries = run_calib_measure_batch_subprocess(
        [(rd, md, ft) for _, rd, md, ft in calib_items],
        device=device,
    )
    return {lvl_tag: batch_summaries[ft] for lvl_tag, _, _, ft in calib_items}


@torch.no_grad()
def evaluate_holdout_v35(
    pcore: PCoreSLFUNet,
    *,
    tag: str,
    fcfg: dict[str, Any],
    knob_s: float = 0.85,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate zero-DINO model on V18 holdout-750 using finalized single-knob schedule."""
    m_dir = OUT / "holdout/measured" / tag
    sum_p = m_dir / "summary_holdout.json"
    if sum_p.exists():
        return read_json(sum_p)

    spec = build_or_load_holdout_spec(device=device)
    tasks = spec["tasks"]
    src_idx = spec["src_idx"]

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ho = v18["holdout"]
    z_src_ho = ho["z"][src_idx].float()
    c64_ho = ho["c_64"][src_idx].float()
    y16_ho = project_latent_y16_lcn(z_src_ho, v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())

    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    reps = build_or_load_content_ablation_reps(device=device)
    q16_ho = reps["holdout"]["stiefel"][src_idx].float()
    ho_gray = stiefel["holdout"].get("cnx_gray")
    if ho_gray is None:
        dec_tmp = load_s10_decoder(device)
        cnx_ext = ConvNeXtTinyStage3Extractor(device=device).eval()
        ho_gray = cnx_ext.extract_gray_layout(decode_s10(dec_tmp, ho["z"][src_idx].to(device).float(), batch=32, output_cpu=False)).cpu()
        del dec_tmp, cnx_ext
    else:
        ho_gray = ho_gray[src_idx]
    cnx_gray_ho = ho_gray.float()

    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    dec = load_s10_decoder(device)
    src_pos = torch.tensor([t["source_pos"] for t in tasks], dtype=torch.long)
    tid_all = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    eps_src = torch.randn(z_src_ho.shape, generator=torch.Generator().manual_seed(EVAL_SEED + 777))

    kw = make_v35_schedule(knob_s, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
    lam = float(kw.get("draft_lambda", 0.75))
    kw_call = {k: v for k, v in kw.items() if k != "draft_lambda"}

    raw_dir = OUT / "holdout/raw" / tag
    writer = RawWriterV35(tag, dec=dec, device=device, raw_root=OUT / "holdout/raw")
    infer_bs = auto_infer_batch(device)
    for st in range(0, len(tasks), infer_bs):
        sl = slice(st, min(st + infer_bs, len(tasks)))
        sp = src_pos[sl]
        ti = tid_all[sl].to(device)
        ze = ecore(
            z_src_ho[sp].to(device), c64_ho[sp].to(device), q16_ho[sp].to(device),
            ti, torch.full((len(ti),), lam, device=device), raw_stats, priors_ec,
            noise=eps_src[sp].to(device),
        )
        ze_start = build_pcore_startpoint(ze, ti, priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
        u_st = match_cnx_k8_submode(cnx_gray_ho[sp].to(device), ti, cnx_bank)
        zo = integrate_v33_trajectory(pcore, ze_start, eps_src[sp].to(device), y16_ho[sp].to(device), q16_ho[sp].to(device), ti, u_st, steps=int(fcfg["steps"]), solver="dpm2m", **kw_call)
        writer.write_batch(zo, tasks[sl])
    writer.finalize(arm=tag, holdout=True, schedule=kw, fcfg=fcfg)

    del ecore, dec
    gc.collect()
    torch.cuda.empty_cache()
    return run_holdout_measure_subprocess(raw_dir, m_dir, arm_name=tag, device=device)


@torch.no_grad()
def measure_e2e_speed_and_params(ckpt_path: Path, fcfg: dict[str, Any], *, weight_key: str = "ema", device: str = "cuda") -> dict[str, Any]:
    """Measure cold-start RGB->RGB latency (b1/b16), stage breakdown, peak VRAM, NFE, and exact parameter ledger."""
    sanitize_cuda_env()
    pcore, _ = load_pcore_v35(ckpt_path, weight_key=weight_key, device=device)
    assets = build_or_load_cnx_assets(device=device)
    w32_st = assets["stiefel"]["W_stiefel_cnx"].to(device).float()
    cnx_bank = assets["bank"]

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    w_luma = v18["luma_proj"]["w"].to(device).float()
    b_luma = v18["luma_proj"]["b"].to(device).float()

    a0_enc = load_a0_encoder(device).to(torch.bfloat16).eval().requires_grad_(False)
    dec = load_s10_decoder(device)
    cnx_ext = ConvNeXtTinyStage3Extractor(device=device).eval()
    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)

    kw = make_v35_schedule(0.85, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
    lam_val = float(kw.pop("draft_lambda", 0.75))
    n_steps = int(fcfg["steps"])

    speed: dict[str, Any] = {}
    breakdown_b16: dict[str, float] = {}
    for bs in (1, 16):
        rgb_in = torch.rand(bs, 3, 512, 512, device=device)
        tid = torch.arange(bs, device=device) % 5
        eps = torch.randn(bs, 4, 64, 64, device=device)
        lam = torch.full((bs,), lam_val, device=device)

        def _timed_e2e(collect_parts: bool = False):
            timings: dict[str, float] = {}
            def _mark(k: str, t_s: float):
                if collect_parts:
                    torch.cuda.synchronize()
                    timings[k] = timings.get(k, 0.0) + (time.perf_counter() - t_s) * 1000.0
                return time.perf_counter()

            t = time.perf_counter()
            z_s = (a0_enc.encode(rgb_in.to(torch.bfloat16)).latent_dist.mean * 0.18215).float()
            y64 = rgb_to_structure_64(rgb_in)
            c64 = torch.cat([y64, local_contrast_norm(rgb_to_gray(F.interpolate(rgb_in, (64, 64), mode="area"))), y64, y64], dim=1)[:, :4]
            y16 = project_latent_y16_lcn(z_s, w_luma, b_luma)
            t = _mark("a0_encode_struct_ms", t)

            g16 = cnx_ext.extract_gray_layout(rgb_in)
            q16 = extract_cnx_q32(g16, w32_st)
            u = match_cnx_k8_submode(g16, tid, cnx_bank)
            t = _mark("convnext_stiefel_ms", t)

            ze = ecore(z_s, c64, q16, tid, lam, raw_stats, priors_ec, noise=eps)
            ze_s = build_pcore_startpoint(ze, tid, priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
            t = _mark("ecore_a3_ms", t)

            zo = integrate_v33_trajectory(pcore, ze_s, eps, y16, q16, tid, u, steps=n_steps, solver="dpm2m", **kw)
            t = _mark("pcore_ode_ms", t)

            rgb_out = decode_s10(dec, zo, batch=bs, output_cpu=False)
            _mark("s10_decode_ms", t)
            return rgb_out, timings

        for _ in range(3):
            _timed_e2e(False)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_rep = 12 if bs == 1 else 6
        accum_parts: dict[str, float] = {}
        for _ in range(n_rep):
            _, parts = _timed_e2e(bs == 16)
            for k, v in parts.items():
                accum_parts[k] = accum_parts.get(k, 0.0) + v
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / (n_rep * bs)
        speed[f"b{bs}_ms_per_img"] = round(ms, 2)
        speed[f"b{bs}_ips"] = round(1000.0 / ms, 1)
        speed[f"b{bs}_peak_vram_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 3), 2)
        if bs == 16:
            breakdown_b16 = {k: round(v / (n_rep * bs), 2) for k, v in accum_parts.items()}

    ledger = {
        "a0_vae_encoder_params": sum(p.numel() for p in a0_enc.encoder.parameters()) + sum(p.numel() for p in a0_enc.quant_conv.parameters()),
        "convnext_s3_params": cnx_ext.n_params(),
        "ecore_params": sum(p.numel() for p in ecore.parameters()),
        "pcore_params": pcore.n_params(),
        "s10_decoder_params": sum(p.numel() for p in dec.parameters()),
        "e2e_nfe": {"ecore": 1, "pcore_steps": n_steps, "pcore_cfg_evals": n_steps * 4, "total_flow_evals": 1 + n_steps * 4},
    }
    ledger["gen_plus_dec_total"] = ledger["ecore_params"] + ledger["pcore_params"] + ledger["s10_decoder_params"]
    ledger["full_e2e_total"] = ledger["gen_plus_dec_total"] + ledger["a0_vae_encoder_params"] + ledger["convnext_s3_params"]

    del pcore, a0_enc, dec, cnx_ext, ecore
    gc.collect()
    torch.cuda.empty_cache()
    return {"speed": speed, "breakdown_b16_ms_per_img": breakdown_b16, "param_ledger": ledger}


def train_44m_ablation(
    arm_name: str,
    run_dir: Path,
    *,
    base_dim: int = 208,
    cond_dim: int = 384,
    content_mode: str = "stiefel",   # "stiefel" | "whiten32" | "pca32"
    pool_c_mode: str = "15pct",      # "15pct" | "0pct_self_match"
    updates: int = 3500,
    lr: float = 3.0e-4,
    batch_size: int = 128,
    seed_offset: int = 0,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train PCoreSLFUNet (44.54M at dim=208, or iso-param 58.04M at dim=240) cleanly under V34/V35 recipe."""
    sanitize_cuda_env()
    seed_all(SEED_V35 + seed_offset)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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
            "loss_curve": sv.get("loss_curve", []),
            "train_min": sv.get("train_min", 0.0),
            "peak_vram_gb": sv.get("peak_vram_gb", 0.0),
        }

    assets = build_or_load_cnx_assets(device=device)
    reps = build_or_load_content_ablation_reps(device=device)
    q16_all = reps["train"][content_mode]
    cnx_bank = assets["bank"]
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

    swd_critic = _HighTEndpointSWDCritic(device=device)
    swd_critic.set_moment_scale(torch.cat([z_all[ci[:10]] for ci in core_cls], dim=0))

    model = PCoreSLFUNet(base_dim=int(base_dim), u_dim=417, cond_dim=int(cond_dim), use_gamma=True, normalize_u=True).to(device)
    cfg = {
        "base_dim": int(base_dim),
        "u_dim": 417,
        "cond_dim": int(cond_dim),
        "use_adapter": False,
        "pcore_params": model.n_params(),
        "content_mode": content_mode,
        "pool_c_mode": pool_c_mode,
        "updates": updates,
        "seed_offset": seed_offset,
    }
    model.train().requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1.0e-4)
    params_list = list(model.parameters())

    start_step = 0
    loss_curve: list[dict[str, float]] = []
    if resume_p.exists() and smoke_steps == 0:
        r_sv = torch.load(resume_p, map_location="cpu", weights_only=False)
        model.load_state_dict(r_sv["online"], strict=True)
        ema.load_state_dict(r_sv["ema"], strict=True)
        opt.load_state_dict(r_sv["opt"])
        start_step = int(r_sv["step"])
        loss_curve = list(r_sv.get("loss_curve", []))
        print(f"[{arm_name}] Resuming from step {start_step}/{updates}", flush=True)

    micro_24gb = 64 if int(base_dim) <= 208 else 32
    micro, accum = tier_micro_accum(batch_size, default_micro_24gb=micro_24gb, max_micro_12gb=32, max_micro_8gb=16, device=device)
    swd_micro = 6 if vram_total_gb(device) > 13.5 else 2
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
        f"[{arm_name}] Start train | params={cfg['pcore_params']/1e6:.2f}M | content={content_mode} "
        f"| pool_c={pool_c_mode} | micro={micro}x{accum} swd_micro={swd_micro}",
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
        r_pc = pc_ridx[p_pick]

        if pool_c_mode == "0pct_self_match":
            # Replace cross-style pairs with self-paired samples of the EXACT SAME target real paintings
            z_pc_src = z_all[r_pc]
            q_pc_src = q16_all[r_pc]
            y_pc_src = y16_all[r_pc]
            t_pc_src = sid_all[r_pc]
        else:
            z_pc_src = pc_zreal[p_pick]
            q_pc_src = pc_q16[p_pick]
            y_pc_src = pc_y16[p_pick]
            t_pc_src = pc_tid[p_pick]

        z1 = torch.cat([z_all[idx_self], z_pc_src], dim=0).to(device, non_blocking=True).float()
        tid = torch.cat([sid_all[idx_self], t_pc_src], dim=0).to(device, non_blocking=True)
        q16 = torch.cat([q16_all[idx_self], q_pc_src], dim=0).to(device, non_blocking=True).float()
        y16 = torch.cat([y16_all[idx_self], y_pc_src], dim=0).to(device, non_blocking=True).float()

        real_indices = torch.cat([idx_self, r_pc], dim=0)
        u_ex = norm417_all[real_indices].to(device, non_blocking=True).float()
        sub_k = torch.randint(0, 8, (batch_size,))
        u_sub = sub_raw417[tid.cpu(), sub_k].to(device, non_blocking=True).float()
        u_in = torch.where(torch.rand(batch_size, 1, device=device) < 0.75, u_ex, u_sub)

        q16_aug = q16 + 0.06 * torch.randn_like(q16)
        q16_aug = q16_aug - q16_aug.mean(dim=(2, 3), keepdim=True)
        y16_aug, y16_clean_m = _blur_or_drop_y16(y16, drop_prob=0.40)

        ds = torch.rand(batch_size, device=device) < 0.15
        dc = torch.rand(batch_size, device=device) < 0.30
        lscale = torch.where(dc, torch.zeros(batch_size, device=device), 0.25 + 0.75 * torch.rand(batch_size, device=device))
        gamma_b = torch.where(torch.rand(batch_size, device=device) < 0.40, torch.ones(batch_size, device=device), 0.15 + 0.85 * torch.rand(batch_size, device=device))

        z0 = torch.randn_like(z1)
        tau = torch.rand(batch_size, device=device)
        zt = (1.0 - tau).view(batch_size, 1, 1, 1) * z0 + tau.view(batch_size, 1, 1, 1) * z1
        target_v = z1 - z0

        phuber_acc, st_acc, pix_val, gn = run_pcore_train_step(
            model=model, opt=opt, ema=ema, params_list=params_list, swd_critic=swd_critic,
            w_luma=w_luma, b_luma=b_luma,
            z1=z1, zt=zt, target_v=target_v,
            y16=y16, y16_aug=y16_aug, y16_clean_m=y16_clean_m, q16_aug=q16_aug,
            tid=tid, u_in=u_in, tau=tau, ds=ds, dc=dc, lscale=lscale, gamma_b=gamma_b,
            micro=micro, accum=accum, swd_micro=swd_micro, step=step, seed_base=SEED_V35,
            use_adapter=False, q32_hi_aug=None, device=device,
        )

        if step == 1 or step % 250 == 0 or step == target_steps:
            dt = (time.monotonic() - t0) / max(1, step - start_step)
            loss_curve.append({"step": step, "phuber": round(phuber_acc, 4), "st": round(st_acc, 4), "pix": round(pix_val, 4), "gn": round(float(gn), 3)})
            if step == 1 or step % 500 == 0 or step == target_steps:
                print(
                    f"[{arm_name}] {step:4d}/{target_steps} phuber={phuber_acc:.4f} st={st_acc:.4f} pix={pix_val:.4f} gn={float(gn):.2f} ({dt:.3f}s/st)",
                    flush=True,
                )

        if smoke_steps == 0 and (step % 500 == 0 and step < target_steps):
            atomic_save({
                "online": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "cfg": cfg,
                "loss_curve": loss_curve,
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
        "loss_curve": loss_curve,
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
        "loss_curve": loss_curve,
        "train_min": elapsed_min,
        "peak_vram_gb": round(vram_gb, 3),
    }
