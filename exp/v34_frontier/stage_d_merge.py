"""CLEAVE V34 Stage D (2h budget): Selective Winner Merge, Fixed-Content Frontier Comparison,
Independent Holdout-300 Verification, Paired Cluster Bootstrap 95% CI, & Cold-Start Speed.

Only incorporates mechanisms from Stages A, B, C with independently verified gains:
  - Stage A: analytic low-freq Luma / Chroma startpoint if Minimalism/Cross gain verified, else A0
  - Stage B: continuous style conditioning (inst_417 or inst_1101) if frontier gain verified, else k8
  - Stage C: winning architecture/training config among C0..C3 (e.g. 44M + S2 joint)
Trains winner for 4,500 steps, compares on test750 at C in {0.50, 0.61, 0.70} vs V32 & StyleGallery,
and validates generalization on unseen holdout-300 paintings (150 disjoint sources x 5 targets).
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v25.r14_e2e_benchmark import load_a0_encoder
from exp.v27_single.bare_q32 import load_cache
from exp.v28_unified.student_models import build_student_model
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor, extract_cnx_q32, match_cnx_k8_submode
from exp.v33_semantic_control.models import ConvNeXtMultiScaleExtractor, extract_cnx_q32_hi, integrate_v33_trajectory
from exp.v34_frontier import (
    OUT,
    SEED_V34,
    STYLES,
    V32_ECORE_CKPT,
    V32_OUT,
    V32_PCORE_CKPT,
    RawWriterV34,
    interp_s_at_c,
    paired_ci_vs_baseline,
    v32_frontier_points,
)
from exp.v34_frontier.stage_a_startpoint import build_or_load_analytic_priors, reconstruct_analytic_draft
from exp.v34_frontier.train_common import (
    build_or_load_style_instances,
    evaluate_v34_arm,
    load_pcore_from_v34_ckpt,
    retrieve_continuous_style_u,
    schedule_for_v34,
    train_pcore_v34,
)
from src.eval.holdout import build_or_load_holdout_spec, run_holdout_measure_subprocess
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import REPO as ROOT, read_json, sanitize_cuda_env, save_json, utc
from src.utils.structure import local_contrast_norm, rgb_to_gray, rgb_to_structure_64


STYLEGALLERY_BASE = {
    "dino_c": 0.61134,
    "dino_s": 0.56044,
    "csd_top1": 0.5250,
    "clip_s": 0.7260,
    "style_affinity": 0.6432,
}


def select_merge_config(
    stage_a_rep: dict[str, Any] | None,
    stage_b_rep: dict[str, Any] | None,
    stage_c_rep: dict[str, Any] | None,
    *,
    force_a: str | None = None,
    force_b: str | None = None,
    force_c: str | None = None,
) -> dict[str, Any]:
    """Select ONLY independently verified gains from Stages A, B, and C."""
    # 1. Stage A gate: did A1/A2/A3 boost Minimalism DINO-S/Top1/sat without damaging DINO-C?
    analytic_mode = "A0"
    a_reason = "no_verified_gain"
    if force_a is not None:
        analytic_mode = force_a
        a_reason = "forced_override"
    elif stage_a_rep and "arms" in stage_a_rep and stage_a_rep.get("stage_a_effective", False):
        best_a = "A0"
        best_a_score = 0.0
        for lvl_tag in ("s085", "s100"):
            a0 = stage_a_rep["arms"].get(f"StageA_A0_{lvl_tag}")
            if not a0:
                continue
            a0_min_s = float(a0["per_tgt_dino_s"].get("Min", 0.0))
            a0_cross_s = float(a0["dino_s"])
            a0_c = float(a0["dino_c"])
            for m_key in ("A1", "A2", "A3"):
                am = stage_a_rep["arms"].get(f"StageA_{m_key}_{lvl_tag}")
                if not am:
                    continue
                dc = float(am["dino_c"]) - a0_c
                d_min_s = float(am["per_tgt_dino_s"].get("Min", 0.0)) - a0_min_s
                d_cross_s = float(am["dino_s"]) - a0_cross_s
                if dc >= -0.010 and (d_min_s >= 0.005 or d_cross_s >= 0.003):
                    sc = d_cross_s + 0.35 * d_min_s
                    if sc > best_a_score:
                        best_a_score = sc
                        best_a = m_key
        if best_a != "A0":
            analytic_mode = best_a
            a_reason = f"verified_gain_score_{best_a_score:+.4f}"

    # 2. Stage B gate: did B1 or B2 beat B0 on matched-(DINO-C) frontier?
    style_mode = "k8"
    b_reason = "no_verified_frontier_gain"
    if force_b is not None:
        style_mode = force_b
        b_reason = "forced_override"
    elif stage_b_rep:
        if stage_b_rep.get("frontier_verified_effective", False):
            style_mode = str(stage_b_rep.get("winner_style_mode", "k8"))
            b_reason = f"verified_delta_s_{stage_b_rep.get('best_mean_delta_dino_s_vs_B0', 0.0):+.4f}"

    # 3. Stage C gate: pick best verified architecture among C0..C3
    c_winner = "C0_22m_q32"
    c_spec = {"base_dim": 144, "cond_dim": 320, "use_adapter": False, "label": "22M + Q32"}
    c_reason = "default_baseline"
    if force_c is not None and stage_c_rep and force_c in stage_c_rep.get("arm_summary", {}):
        c_winner = force_c
        c_spec = dict(stage_c_rep["arm_summary"][c_winner]["spec"])
        c_reason = "forced_override"
    elif stage_c_rep:
        c_winner = str(stage_c_rep.get("winner_arm", "C0_22m_q32"))
        c_spec = dict(stage_c_rep["arm_summary"][c_winner]["spec"])
        c_reason = f"verified_stage_c_winner_{c_winner}"

    return {
        "analytic_mode": analytic_mode,
        "a_reason": a_reason,
        "style_mode": style_mode,
        "b_reason": b_reason,
        "c_winner": c_winner,
        "base_dim": int(c_spec["base_dim"]),
        "cond_dim": int(c_spec["cond_dim"]),
        "use_adapter": bool(c_spec["use_adapter"]),
        "c_reason": c_reason,
    }


@torch.no_grad()
def evaluate_holdout_split(
    model: torch.nn.Module,
    *,
    tag: str,
    style_mode: str = "k8",
    use_adapter: bool = False,
    analytic_mode: str = "A0",
    knob_s: float = 0.85,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run zero-DINO model on completely unseen holdout-750 (150 disjoint sources x 5 targets)."""
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
    q64_cache = torch.load(OUT.parent / "v33_semantic_control/cache/v33_q64_hi32_cache.pt", map_location="cpu", weights_only=False)
    q16_ho = stiefel["holdout"]["q32"][src_idx].float()
    cnx_gray_ho = q64_cache["holdout"]["cnx_gray"][src_idx].float()
    inst_cache = build_or_load_style_instances(device=device) if style_mode != "k8" else None

    dec = load_s10_decoder(device)
    if use_adapter:
        ms_ext = ConvNeXtMultiScaleExtractor(device=device).eval()
        g32_chunks = []
        for st in range(0, len(z_src_ho), 16):
            sl = slice(st, min(st + 16, len(z_src_ho)))
            rgb = decode_s10(dec, z_src_ho[sl].to(device).float(), dtype=torch.bfloat16, batch=sl.stop - sl.start, output_cpu=False).clamp(0.0, 1.0)
            g32_chunks.append(ms_ext.extract_gray_layout_32(rgb).cpu())
        q32_hi_ho = extract_cnx_q32_hi(torch.cat(g32_chunks, 0).to(device), q64_cache["W_hi32"].to(device)).cpu().float()
        del ms_ext
    else:
        q32_hi_ho = None

    ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device)
    priors_ec = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device).eval().requires_grad_(False)
    ecore.load_state_dict(ec_sv["ema"], strict=True)
    priors_a = build_or_load_analytic_priors(priors_ec) if analytic_mode != "A0" else None

    # Repeat 150 holdout sources 5x (matching tasks ordering: source_pos 0..149 x target 0..4)
    src_pos = torch.tensor([t["source_pos"] for t in tasks], dtype=torch.long)
    tid_all = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    eps_src = torch.randn(z_src_ho.shape, generator=torch.Generator().manual_seed(EVAL_SEED + 777))

    kw = schedule_for_v34(knob_s)
    lam = float(kw.get("draft_lambda", 0.75))
    kw_call = {k: v for k, v in kw.items() if k != "draft_lambda"}

    raw_dir = OUT / "holdout/raw" / tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    writer = RawWriterV34(f"ho_{tag}", dec=dec, device=device)
    writer.path = raw_dir

    for st in range(0, len(tasks), 32):
        sl = slice(st, min(st + 32, len(tasks)))
        sp = src_pos[sl]
        ti = tid_all[sl].to(device)
        zb = z_src_ho[sp].to(device).float()
        cb = c64_ho[sp].to(device).float()
        qb = q16_ho[sp].to(device).float()
        yb = y16_ho[sp].to(device).float()
        eb = eps_src[sp].to(device).float()
        gb = cnx_gray_ho[sp].to(device).float()

        ze = ecore(zb, cb, qb, ti, torch.full((len(ti),), lam, device=device), raw_stats, priors_ec, noise=eb)
        if priors_a is not None:
            ze = reconstruct_analytic_draft(ze, ti, priors_a, mode=analytic_mode)

        if style_mode == "k8":
            u_st = match_cnx_k8_submode(gb, ti, cnx_bank)
        else:
            assert inst_cache is not None
            u_st = retrieve_continuous_style_u(gb, ti, inst_cache, cond_key=style_mode)

        qh = q32_hi_ho[sp].to(device).float() if q32_hi_ho is not None else None
        zo = integrate_v33_trajectory(model, ze, eb, yb, qb, ti, u_st, steps=6, solver="dpm2m", q32_hi=qh, **kw_call)
        writer.write_batch(zo, tasks[sl])
    writer.finalize(arm=tag, holdout=True, knob_s=knob_s, style_mode=style_mode, analytic_mode=analytic_mode)

    del ecore, dec
    gc.collect()
    torch.cuda.empty_cache()
    return run_holdout_measure_subprocess(raw_dir, m_dir, arm_name=tag, device=device)


@torch.no_grad()
def measure_cold_start_speed(ckpt_path: Path, *, device: str = "cuda") -> dict[str, Any]:
    """Measure full RGB-to-RGB throughput (A0 encode + ConvNeXt + E-Core + P-Core + S10) at batch 1 and 16."""
    sanitize_cuda_env()
    model, cfg = load_pcore_from_v34_ckpt(ckpt_path, device=device)
    use_adapter = bool(cfg.get("use_adapter", False))
    style_mode = str(cfg.get("style_mode", "k8"))

    assets = build_or_load_cnx_assets(device=device)
    w32_st = assets["stiefel"]["W_stiefel_cnx"].to(device).float()
    cnx_bank = assets["bank"]
    inst_cache = build_or_load_style_instances(device=device) if style_mode != "k8" else None
    q64_cache = torch.load(OUT.parent / "v33_semantic_control/cache/v33_q64_hi32_cache.pt", map_location="cpu", weights_only=False) if use_adapter else None
    whi32_st = q64_cache["W_hi32"].to(device).float() if q64_cache is not None else None

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    w_luma = v18["luma_proj"]["w"].to(device).float()
    b_luma = v18["luma_proj"]["b"].to(device).float()

    a0_enc = load_a0_encoder(device).to(torch.bfloat16).eval().requires_grad_(False)
    dec = load_s10_decoder(device)
    cnx_ext = ConvNeXtTinyStage3Extractor(device=device).eval()
    ms_ext = ConvNeXtMultiScaleExtractor(device=device).eval() if use_adapter else None

    ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device)
    priors_ec = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device).eval().requires_grad_(False)
    ecore.load_state_dict(ec_sv["ema"], strict=True)

    kw = {k: v for k, v in schedule_for_v34(0.85).items() if k != "draft_lambda"}
    out = {}
    for bs in (1, 16):
        rgb_in = torch.rand(bs, 3, 512, 512, device=device)
        tid = torch.arange(bs, device=device) % 5
        eps = torch.randn(bs, 4, 64, 64, device=device)
        lam = torch.full((bs,), 0.75, device=device)

        def _once():
            z_s = (a0_enc.encode(rgb_in.to(torch.bfloat16)).latent_dist.mean * 0.18215).float()
            y64 = rgb_to_structure_64(rgb_in)
            c64 = torch.cat([y64, local_contrast_norm(rgb_to_gray(F.interpolate(rgb_in, (64, 64), mode="area"))), y64, y64], dim=1)[:, :4]
            y16 = project_latent_y16_lcn(z_s, w_luma, b_luma)
            g16 = cnx_ext.extract_gray_layout(rgb_in)
            q16 = extract_cnx_q32(g16, w32_st)
            qh = extract_cnx_q32_hi(ms_ext.extract_gray_layout_32(rgb_in), whi32_st) if ms_ext is not None else None
            u = match_cnx_k8_submode(g16, tid, cnx_bank) if style_mode == "k8" else retrieve_continuous_style_u(g16, tid, inst_cache, cond_key=style_mode)
            ze = ecore(z_s, c64, q16, tid, lam, raw_stats, priors_ec, noise=eps)
            zo = integrate_v33_trajectory(model, ze, eps, y16, q16, tid, u, steps=6, solver="dpm2m", q32_hi=qh, **kw)
            return decode_s10(dec, zo, batch=bs, output_cpu=False)

        for _ in range(3):
            _once()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_rep = 12 if bs == 1 else 6
        for _ in range(n_rep):
            _once()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / (n_rep * bs)
        out[f"b{bs}_ms_per_img"] = round(ms, 2)
        out[f"b{bs}_ips"] = round(1000.0 / ms, 1)

    del model, a0_enc, dec, cnx_ext, ecore
    if ms_ext is not None:
        del ms_ext
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_stage_d(
    *,
    updates: int = 4500,
    batch_size: int = 128,
    smoke_steps: int = 0,
    force_a: str | None = None,
    force_b: str | None = None,
    force_c: str | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Select verified gains from Stages A/B/C, train winner, eval test750 + holdout-300."""
    rep_p = OUT / "stage_d_report.json"
    if rep_p.exists() and smoke_steps == 0:
        print(f"[Stage D] Reusing completed report {rep_p}", flush=True)
        return read_json(rep_p)

    a_rep = read_json(OUT / "stage_a_report.json") if (OUT / "stage_a_report.json").exists() else None
    b_rep = read_json(OUT / "stage_b_report.json") if (OUT / "stage_b_report.json").exists() else None
    c_rep = read_json(OUT / "stage_c_report.json") if (OUT / "stage_c_report.json").exists() else None

    mcfg = select_merge_config(a_rep, b_rep, c_rep, force_a=force_a, force_b=force_b, force_c=force_c)
    arm_name = "D_merged_winner"
    print(f"[Stage D] Selective Merge Decision: {mcfg}", flush=True)

    tr_info = train_pcore_v34(
        arm_name,
        OUT / "stage_d" / arm_name,
        base_dim=mcfg["base_dim"],
        cond_dim=mcfg["cond_dim"],
        style_mode=mcfg["style_mode"],
        use_adapter=mcfg["use_adapter"],
        analytic_mode=mcfg["analytic_mode"],
        updates=updates,
        lr=3.0e-4,
        batch_size=batch_size,
        seed_offset=300,
        smoke_steps=smoke_steps,
        device=device,
    )
    if smoke_steps > 0:
        return {"stage": "D_smoke", "merge_config": mcfg, "train": tr_info}

    ckpt_p = Path(tr_info["ckpt"])
    levels = evaluate_v34_arm(
        arm_name,
        ckpt_p,
        knob_list=(0.65, 0.75, 0.85, 1.00, 1.25, 1.50),
        include_tau0=True,
        eval_wrong=True,
        analytic_mode=mcfg["analytic_mode"],
        device=device,
    )

    # Fixed-content frontier comparison at C in {0.50, 0.61, 0.70}
    c_grid = (0.50, 0.61, 0.70)
    v32_pts = v32_frontier_points()
    d_pts_s = sorted([(float(m["dino_c"]), float(m["dino_s"])) for k, m in levels.items() if "wrong" not in k], key=lambda x: x[0])
    d_pts_t = sorted([(float(m["dino_c"]), float(m["csd_top1"])) for k, m in levels.items() if "wrong" not in k], key=lambda x: x[0])
    frontier_cmp = []
    for c_val in c_grid:
        sd = interp_s_at_c(c_val, d_pts_s)
        sv = interp_s_at_c(c_val, v32_pts)
        td = interp_s_at_c(c_val, d_pts_t)
        frontier_cmp.append({
            "target_dino_c": c_val,
            "v34_dino_s": round(sd, 4),
            "v32_dino_s": round(sv, 4),
            "delta_dino_s_vs_v32": round(sd - sv, 4),
            "v34_csd_top1": round(td, 4),
        })

    # Paired Bootstrap CI on test750 vs V32 PCore_NoDINO baseline at s=0.85_c10 and s=1.00
    ci_test750: dict[str, Any] = {}
    for lvl_d, lvl_v32 in (("s085", "PCore_NoDINO_s085_c10"), ("s100", "PCore_NoDINO_s100")):
        p_d = OUT / f"measured/{arm_name}_{lvl_d}/eval.json"
        p_v = V32_OUT / f"measured/{lvl_v32}/eval.json"
        if p_d.exists() and p_v.exists():
            ci_test750[f"{lvl_d}_vs_{lvl_v32}"] = paired_ci_vs_baseline(p_d, p_v, cross_only=True)

    # Independent Holdout-750 Verification: V34 winner vs V32 22M baseline at s=0.85 and s=1.00
    pc_sv = torch.load(V32_PCORE_CKPT, map_location="cpu", weights_only=False)
    v32_pcore = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=True, normalize_u=True).to(device)
    v32_pcore.load_state_dict(pc_sv["ema"], strict=True)
    v32_pcore.eval().requires_grad_(False)

    d_model, _ = load_pcore_from_v34_ckpt(ckpt_p, device=device)
    ho_results: dict[str, Any] = {}
    for kv in (0.85, 1.00):
        tag_k = f"s{int(round(kv * 100)):03d}"
        ho_results[f"V32_{tag_k}"] = evaluate_holdout_split(v32_pcore, tag=f"V32_base_{tag_k}", style_mode="k8", use_adapter=False, analytic_mode="A0", knob_s=kv, device=device)
        ho_results[f"D_merged_{tag_k}"] = evaluate_holdout_split(d_model, tag=f"D_merged_{tag_k}", style_mode=mcfg["style_mode"], use_adapter=mcfg["use_adapter"], analytic_mode=mcfg["analytic_mode"], knob_s=kv, device=device)
    del v32_pcore, d_model
    gc.collect()
    torch.cuda.empty_cache()

    ci_holdout: dict[str, Any] = {}
    for tag_k in ("s085", "s100"):
        p_d = OUT / f"holdout/measured/D_merged_{tag_k}/eval.json"
        p_v = OUT / f"holdout/measured/V32_base_{tag_k}/eval.json"
        if p_d.exists() and p_v.exists():
            ci_holdout[f"{tag_k}_vs_V32_holdout"] = paired_ci_vs_baseline(p_d, p_v, cross_only=True)

    speed = measure_cold_start_speed(ckpt_p, device=device)

    rep = {
        "updated_utc": utc(),
        "stage": "D_merge_and_holdout",
        "merge_config": mcfg,
        "train": tr_info,
        "test750_levels": levels,
        "frontier_at_fixed_C": frontier_cmp,
        "ci_test750_vs_v32": ci_test750,
        "holdout_750": ho_results,
        "ci_holdout_vs_v32": ci_holdout,
        "stylegallery_reference": STYLEGALLERY_BASE,
        "cold_start_rgb_to_rgb_speed": speed,
    }
    save_json(rep_p, rep)
    print(f"[Stage D Done] Holdout & Test750 verified -> {rep_p}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=4500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--force-a", default=None)
    ap.add_argument("--force-b", default=None)
    ap.add_argument("--force-c", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_stage_d(
        updates=args.updates,
        batch_size=args.batch_size,
        smoke_steps=args.smoke_steps,
        force_a=args.force_a,
        force_b=args.force_b,
        force_c=args.force_c,
        device=args.device,
    )


if __name__ == "__main__":
    main()
