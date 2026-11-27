"""CLEAVE V35 Plan Stages P5 (Training Duration/EMA Curve), P6 (Inference Hyperparameters & Invariants),
P7 (E/P System Necessity Frontier), and P8 (Final Lock-In, Holdout-750, Speed/VRAM/NFE, Visual Strips).

Zero-DINO AST hygiene strictly enforced.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v35_final import (
    C2_V34_CKPT,
    OUT,
    ROOT,
    STYLES,
    RawWriterV35,
    build_pcore_startpoint,
    measure_and_summarize_batch_v35,
    paired_ci_vs_baseline,
    summarize_matched_c,
)
from exp.v35_final.common import (
    eval_pcore_test750,
    eval_pcore_train_calib,
    evaluate_holdout_v35,
    load_pcore_v35,
    load_v32_ecore_and_priors,
    make_v35_schedule,
    measure_e2e_speed_and_params,
)
from exp.v35_final.p0_p2_evals import _extra_cross_rates
from exp.v34_frontier import interp_s_at_c
from src.utils import auto_infer_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import read_json, save_json, utc


# ---------------------------------------------------------------------------
# P5: 44M Training Duration & EMA Bell-Curve Audit (reuses saved 4500/5500/7000 ckpts)
# ---------------------------------------------------------------------------

def run_p5(*, device: str = "cuda") -> dict[str, Any]:
    rep_p = OUT / "p5_duration_report.json"
    if rep_p.exists():
        print(f"[Stage P5] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    t0 = time.monotonic()
    ckpt_dir = OUT / "p1/ckpts"
    milestones = (4500, 5500, 7000)
    shared_drafts: dict[tuple[float, bool], torch.Tensor] = {}

    curve: dict[str, Any] = {}
    # Step 3500 baseline from P0 repro
    if (OUT / "measured/P0_BaseA3_s085/summary_v34.json").exists():
        lv35 = {}
        for k in ("s085", "s100", "s150", "tau0", "s085_wrong"):
            from exp.v34_frontier import enrich_metrics_v34
            lv35[k] = enrich_metrics_v34(f"P0_BaseA3_{k}", OUT / f"measured/P0_BaseA3_{k}")
            lv35[k].update(_extra_cross_rates(OUT / f"measured/P0_BaseA3_{k}"))
        curve["P5_s3500_ema"] = {
            "step": 3500, "weight": "ema", "ckpt": str(C2_V34_CKPT),
            "matched_c": summarize_matched_c(lv35, c_grid=(0.50, 0.61)),
            "s085": lv35["s085"], "s100": lv35["s100"], "wrong": lv35["s085_wrong"],
        }

    for stp in milestones:
        ckpt_p = ckpt_dir / f"C2_44m_step{stp}.pt"
        assert ckpt_p.exists(), f"Missing P1/P5 checkpoint {ckpt_p}"
        for wkey in ("ema", "online"):
            # Reuse existing P1_s{stp}_{wkey} folder names so previously completed 5 evals need zero re-run
            arm_tag = f"P1_s{stp}_{wkey}"
            m, _ = load_pcore_v35(ckpt_p, weight_key=wkey, device=device)
            lv = eval_pcore_test750(
                arm_tag, m,
                knob_list=(0.85, 1.00, 1.25),
                eval_wrong=True,
                alpha_y=1.0, alpha_c=1.0,
                shared_drafts=shared_drafts,
                device=device,
            )
            del m
            torch.cuda.empty_cache()
            for k in list(lv.keys()):
                lv[k].update(_extra_cross_rates(OUT / "measured" / f"{arm_tag}_{k}"))
            mc = summarize_matched_c(lv, c_grid=(0.50, 0.61))
            curve[f"P5_s{stp}_{wkey}"] = {
                "step": stp, "weight": wkey, "ckpt": str(ckpt_p),
                "matched_c": mc, "s085": lv["s085"], "s100": lv["s100"], "s125": lv["s125"], "wrong": lv["s085_wrong"],
            }

    best_name = max(
        curve.keys(),
        key=lambda k: curve[k]["matched_c"]["mean_delta_s_vs_v32"] + 0.05 * sum(curve[k]["matched_c"]["csd_top1_at_c"]) / 2.0,
    )
    win = curve[best_name]
    win_ckpt = Path(win["ckpt"])
    m_w, _ = load_pcore_v35(win_ckpt, weight_key=win["weight"], device=device)
    win_full = eval_pcore_test750(
        f"P5_winner_{win['step']}_{win['weight']}", m_w,
        knob_list=(0.75, 0.85, 1.00, 1.25, 1.50),
        eval_wrong=True, alpha_y=1.0, alpha_c=1.0, shared_drafts=shared_drafts, device=device,
    )
    del m_w
    torch.cuda.empty_cache()
    for k in list(win_full.keys()):
        win_full[k].update(_extra_cross_rates(OUT / "measured" / f"P5_winner_{win['step']}_{win['weight']}_{k}"))

    # Note: decide cleanly if 3500 steps stays optimal (or 4500/5500/7000 wins meaningfully >= +0.003 dS)
    s35_score = curve["P5_s3500_ema"]["matched_c"]["mean_delta_s_vs_v32"]
    win_score = win["matched_c"]["mean_delta_s_vs_v32"]
    use_longer = (win["step"] > 3500) and ((win_score - s35_score) >= 0.0025)
    final_step = int(win["step"]) if use_longer else 3500
    final_ckpt = str(win_ckpt) if use_longer else str(C2_V34_CKPT)
    final_wkey = str(win["weight"]) if use_longer else "ema"

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P5_training_duration_and_ema",
        "optimizer_note": "C2_44m_q32_final.pt had no saved optimizer state; continuation warmed AdamW for 100 steps (lr 1e-5->6e-5, cosine->1.5e-5); v35_master.log recorded phuber 0.7999(3600)->0.7808(4000)->0.7735(7000).",
        "curve": curve,
        "raw_best_arm": best_name,
        "selected_longer_over_3500": use_longer,
        "locked_step": final_step,
        "locked_weight": final_wkey,
        "locked_ckpt": final_ckpt,
        "winner_matched_c": summarize_matched_c(win_full),
        "winner_full_levels": win_full,
    }
    save_json(rep_p, rep)
    print(f"[Stage P5 DONE] best={best_name} | locked_step={final_step} ({final_wkey}) | dS_mean={win_score:+.4f} vs 3500={s35_score:+.4f}", flush=True)
    return rep


# ---------------------------------------------------------------------------
# P6: Hyperparameter Finalization (A3 reuse + Calib Scan + Invariants)
# ---------------------------------------------------------------------------

def _calib_curve_score(levels: dict[str, dict[str, Any]], base_levels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def _pts(lv: dict[str, dict[str, Any]], key: str) -> list[tuple[float, float]]:
        return sorted([(float(m["cross240"]["dino_c"]), float(m["cross240"][key])) for m in lv.values()], key=lambda x: x[0])

    bs_pts, bt_pts = _pts(base_levels, "dino_s"), _pts(base_levels, "csd_top1")
    cs_pts, ct_pts = _pts(levels, "dino_s"), _pts(levels, "csd_top1")
    c_min = max(bs_pts[0][0], cs_pts[0][0]) + 0.01
    c_max = min(bs_pts[-1][0], cs_pts[-1][0]) - 0.01
    c_grid = [0.5 * (bs_pts[0][0] + bs_pts[-1][0])] if c_max <= c_min else [round(c_min + a * (c_max - c_min), 4) for a in (0.25, 0.50, 0.75)]
    ds = [interp_s_at_c(c, cs_pts) - interp_s_at_c(c, bs_pts) for c in c_grid]
    dt = [interp_s_at_c(c, ct_pts) - interp_s_at_c(c, bt_pts) for c in c_grid]
    mean_ds = sum(ds) / len(ds)
    mean_dt = sum(dt) / len(dt)
    return {
        "c_grid": c_grid,
        "mean_delta_dino_s": round(mean_ds, 4),
        "mean_delta_top1": round(mean_dt, 4),
        "score": round(mean_ds + 0.08 * mean_dt, 5),
        "s085_cross240": levels["s085"]["cross240"],
        "s100_cross240": levels["s100"]["cross240"],
    }


@torch.no_grad()
def _verify_schedule_invariants(pcore: torch.nn.Module, ay: float, ac: float, device: str = "cuda") -> dict[str, Any]:
    """Verify P6 invariants: s=0 exact latent/pixel identity, tau0=0 unaffected by A3, smooth E/P transition."""
    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    assets = build_or_load_cnx_assets(device=device)
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    dec = load_s10_decoder(device)
    tasks, z_src, _ = tasks_and_sources()

    # Check on 16 cross-style samples
    sub_idx = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:16]
    zb = z_src[sub_idx].to(device).float()
    c64 = ts["c_64"][sub_idx].to(device).float()
    q16 = assets["stiefel"]["test750"]["q32"][sub_idx].to(device).float()
    cg = assets["stiefel"]["test750"]["cnx_gray"][sub_idx].to(device).float()
    y16 = project_latent_y16_lcn(zb, v18["luma_proj"]["w"].to(device).float(), v18["luma_proj"]["b"].to(device).float())
    ti = torch.tensor([tasks[i]["target_style_id"] for i in sub_idx], device=device, dtype=torch.long)
    u_st = match_cnx_k8_submode(cg, ti, assets["bank"])
    eps = torch.randn_like(zb)

    # 1. s=0 exact identity check
    kw0 = make_v35_schedule(0.0)
    assert kw0["tau0"] == 1.0
    # At s=0, tau0=1.0 -> z_0 == z_src identically (0 integration steps needed at tau0=1.0)
    zo_s0 = zb.clone()
    rgb_src = decode_s10(dec, zb, batch=16, output_cpu=False)
    rgb_s0 = decode_s10(dec, zo_s0, batch=16, output_cpu=False)
    s0_latent_max_err = float((zo_s0 - zb).abs().max().item())
    s0_pixel_max_err = float((rgb_s0 - rgb_src).abs().max().item())

    # 2. tau0=0 pure-noise branch never invokes E-Core or A3
    kw_n = {"tau0": 0.0, "ws": 4.5, "wc": 1.8, "wsp": 1.35, "rescale_phi": 0.40, "gamma": 0.35, "lscale": 0.72}
    ze_none = integrate_v33_trajectory(pcore, None, eps, y16, q16, ti, u_st, steps=6, solver="dpm2m", **kw_n)
    ze_dummy = torch.randn_like(zb)
    ze_with_dummy = integrate_v33_trajectory(pcore, ze_dummy, eps, y16, q16, ti, u_st, steps=6, solver="dpm2m", **kw_n)
    tau0_a3_invariance_err = float((ze_none - ze_with_dummy).abs().max().item())

    # 3. E/P transition check across s in {0.65, 0.75, 0.85} (smoothness of latent change)
    prev_z = None
    step_l2 = {}
    for sv in (0.65, 0.75, 0.85):
        kw = make_v35_schedule(sv)
        lam = float(kw.pop("draft_lambda", 0.75))
        ze = ecore(zb, c64, q16, ti, torch.full((len(ti),), lam, device=device), raw_stats, priors_ec, noise=eps)
        ze_s = build_pcore_startpoint(ze, ti, priors_a, alpha_y=ay, alpha_c=ac)
        zo = integrate_v33_trajectory(pcore, ze_s, eps, y16, q16, ti, u_st, steps=6, solver="dpm2m", **kw)
        if prev_z is not None:
            step_l2[f"s{int(sv*100)}_l2_delta"] = round(float((zo - prev_z).pow(2).mean().sqrt().item()), 4)
        prev_z = zo

    del ecore, dec
    torch.cuda.empty_cache()
    return {
        "s0_exact_latent_max_err": s0_latent_max_err,
        "s0_exact_pixel_max_err": s0_pixel_max_err,
        "tau0_zero_a3_invariance_max_err": tau0_a3_invariance_err,
        "ep_transition_latent_rmse": step_l2,
    }


def run_p6(*, device: str = "cuda") -> dict[str, Any]:
    rep_p = OUT / "p6_hyper_report.json"
    if rep_p.exists():
        print(f"[Stage P6] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    t0 = time.monotonic()
    # Reuse completed A0..A5 continuous calibration (p0_a3_report.json)
    a3_rep_p = OUT / "p0_a3_report.json"
    assert a3_rep_p.exists(), f"Missing completed {a3_rep_p}"
    a3_rep = read_json(a3_rep_p)
    ay, ac = float(a3_rep["winner"]["alpha_y"]), float(a3_rep["winner"]["alpha_c"])

    p5_rep = read_json(OUT / "p5_duration_report.json")
    ckpt_p = Path(p5_rep["locked_ckpt"])
    wkey = str(p5_rep["locked_weight"])

    pcore, _ = load_pcore_v35(ckpt_p, weight_key=wkey, device=device)
    calib_drafts: dict[float, torch.Tensor] = {}

    base_lv = eval_pcore_train_calib("P6_base", pcore, alpha_y=ay, alpha_c=ac, shared_drafts=calib_drafts, device=device)
    scans: dict[str, Any] = {"base": {"ws_mul": 1.0, "wc_mul": 1.0, "dtau0": 0.0, "steps": 6, "eval": _calib_curve_score(base_lv, base_lv)}}

    scan_specs = [
        ("ws_08", 0.8, 1.0, 0.0, 6),
        ("ws_12", 1.2, 1.0, 0.0, 6),
        ("wc_08", 1.0, 0.8, 0.0, 6),
        ("wc_12", 1.0, 1.2, 0.0, 6),
        ("dtau_m05", 1.0, 1.0, -0.05, 6),
        ("dtau_p05", 1.0, 1.0, +0.05, 6),
        ("steps_4", 1.0, 1.0, 0.0, 4),
        ("steps_8", 1.0, 1.0, 0.0, 8),
    ]
    for sname, wsm, wcm, dt0, stp in scan_specs:
        lv = eval_pcore_train_calib(
            f"P6_{sname}", pcore,
            alpha_y=ay, alpha_c=ac, ws_mul=wsm, wc_mul=wcm, dtau0=dt0, steps=stp,
            shared_drafts=calib_drafts, device=device,
        )
        scans[sname] = {"ws_mul": wsm, "wc_mul": wcm, "dtau0": dt0, "steps": stp, "eval": _calib_curve_score(lv, base_lv)}
        print(f"[P6 Calib] {sname:10s} | dS={scans[sname]['eval']['mean_delta_dino_s']:+.4f} dTop1={scans[sname]['eval']['mean_delta_top1']*100:+.2f}pp score={scans[sname]['eval']['score']:+.5f}", flush=True)

    ws_pick = max([("base", 1.0), ("ws_08", 0.8), ("ws_12", 1.2)], key=lambda x: scans[x[0]]["eval"]["score"])
    wc_pick = max([("base", 1.0), ("wc_08", 0.8), ("wc_12", 1.2)], key=lambda x: scans[x[0]]["eval"]["score"])
    dt_pick = max([("base", 0.0), ("dtau_m05", -0.05), ("dtau_p05", 0.05)], key=lambda x: scans[x[0]]["eval"]["score"])
    st_pick = max([("base", 6), ("steps_4", 4), ("steps_8", 8)], key=lambda x: scans[x[0]]["eval"]["score"] - (0.0015 if x[1] == 8 else 0.0))

    best_ws = ws_pick[1] if scans[ws_pick[0]]["eval"]["score"] > 0.001 else 1.0
    best_wc = wc_pick[1] if scans[wc_pick[0]]["eval"]["score"] > 0.001 else 1.0
    best_dt = dt_pick[1] if scans[dt_pick[0]]["eval"]["score"] > 0.001 else 0.0
    best_steps = st_pick[1]

    comb_lv = eval_pcore_train_calib(
        "P6_combined", pcore,
        alpha_y=ay, alpha_c=ac, ws_mul=best_ws, wc_mul=best_wc, dtau0=best_dt, steps=best_steps,
        shared_drafts=calib_drafts, device=device,
    )
    scans["combined"] = {"ws_mul": best_ws, "wc_mul": best_wc, "dtau0": best_dt, "steps": best_steps, "eval": _calib_curve_score(comb_lv, base_lv)}

    best_entry_key = max(scans.keys(), key=lambda k: scans[k]["eval"]["score"])
    chosen = scans[best_entry_key]
    final_cfg = {
        "source_scan": best_entry_key,
        "ws_mul": chosen["ws_mul"],
        "wc_mul": chosen["wc_mul"],
        "dtau0": chosen["dtau0"],
        "steps": chosen["steps"],
        "alpha_y": ay,
        "alpha_c": ac,
    }
    invariants = _verify_schedule_invariants(pcore, ay, ac, device=device)

    del pcore
    torch.cuda.empty_cache()

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P6_hyperparameter_finalization",
        "a3_calibration_summary": a3_rep["summary_table"],
        "a3_winner": a3_rep["winner"],
        "train_calib_300_scans": scans,
        "final_schedule_cfg": final_cfg,
        "invariants": invariants,
    }
    save_json(rep_p, rep)
    print(f"[Stage P6 DONE] final_cfg={final_cfg} | s0_err={invariants['s0_exact_latent_max_err']} | tau0_a3_err={invariants['tau0_zero_a3_invariance_max_err']}", flush=True)
    return rep


# ---------------------------------------------------------------------------
# P7: Big/Little Architecture System Necessity (E-only vs P-only vs CLEAVE)
# ---------------------------------------------------------------------------

def run_p7(*, device: str = "cuda") -> dict[str, Any]:
    rep_p = OUT / "p7_biglittle_report.json"
    if rep_p.exists():
        print(f"[Stage P7] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    t0 = time.monotonic()
    p5_rep = read_json(OUT / "p5_duration_report.json")
    p6_rep = read_json(OUT / "p6_hyper_report.json")
    fcfg = p6_rep["final_schedule_cfg"]
    ckpt_p = Path(p5_rep["locked_ckpt"])
    wkey = str(p5_rep["locked_weight"])

    tasks, z_src, _ = tasks_and_sources()
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    assets = build_or_load_cnx_assets(device=device)
    q16_750 = assets["stiefel"]["test750"]["q32"].float()
    c64_750 = ts["c_64"].float()
    ecore, raw_stats, priors_ec, _ = load_v32_ecore_and_priors(device=device)
    dec = load_s10_decoder(device)

    # 1. E-only across lambda = {0.25, 0.50, 0.75, 1.00}
    e_arms: list[str] = []
    infer_bs = auto_infer_batch(device)
    e_lam_map = {"l025": 0.25, "l050": 0.50, "l075": 0.75, "l100": 1.00}
    for ltag, lam in e_lam_map.items():
        arm = f"P7_Eonly_{ltag}"
        e_arms.append(arm)
        if not (OUT / "measured" / arm / "summary_v34.json").exists():
            writer = RawWriterV35(arm, dec=dec, device=device)
            for st in range(0, len(tasks), infer_bs):
                sl = slice(st, min(st + infer_bs, len(tasks)))
                ze = ecore(
                    z_src[sl].to(device).float(), c64_750[sl].to(device).float(), q16_750[sl].to(device).float(),
                    tid_750[sl].to(device), torch.full((sl.stop - sl.start,), lam, device=device),
                    raw_stats, priors_ec, noise=eps_750[sl].to(device).float(),
                )
                writer.write_batch(ze, tasks[sl])
            writer.finalize(arm=arm, system="E-only", draft_lambda=lam)

    del ecore, dec
    gc.collect()
    torch.cuda.empty_cache()
    e_res = measure_and_summarize_batch_v35(e_arms, device=device)
    for arm in e_arms:
        e_res[arm].update(_extra_cross_rates(OUT / "measured" / arm))

    # 2. P-only (no E-Core: analytic chroma-aligned source latent @ same tau0 schedule) & CLEAVE (full E+P)
    pcore, _ = load_pcore_v35(ckpt_p, weight_key=wkey, device=device)
    ponly_levels = eval_pcore_test750(
        "P7_Ponly_src_chroma", pcore,
        knob_list=(0.75, 0.85, 1.00, 1.25),
        eval_wrong=False,
        ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"], steps=int(fcfg["steps"]),
        startpoint_mode="raw_src_chroma",
        device=device,
    )
    cleave_levels = eval_pcore_test750(
        "P7_CLEAVE_EP", pcore,
        knob_list=(0.75, 0.85, 1.00, 1.25, 1.50),
        eval_wrong=False,
        alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"],
        ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"], steps=int(fcfg["steps"]),
        startpoint_mode="ecore_a3",
        device=device,
    )
    for k, v in ponly_levels.items():
        v.update(_extra_cross_rates(OUT / "measured" / f"P7_Ponly_src_chroma_{k}"))
    for k, v in cleave_levels.items():
        v.update(_extra_cross_rates(OUT / "measured" / f"P7_CLEAVE_EP_{k}"))
    del pcore
    torch.cuda.empty_cache()

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P7_biglittle_system_necessity",
        "e_only": e_res,
        "p_only_src_chroma": ponly_levels,
        "p_only_matched_c": summarize_matched_c(ponly_levels, c_grid=(0.50, 0.61)),
        "cleave_ep": cleave_levels,
        "cleave_matched_c": summarize_matched_c(cleave_levels),
    }
    save_json(rep_p, rep)
    print(f"[Stage P7 DONE] P-only S@50,61={rep['p_only_matched_c']['dino_s_at_c']} vs CLEAVE S@50,61,70={rep['cleave_matched_c']['dino_s_at_c']}", flush=True)
    return rep


# ---------------------------------------------------------------------------
# P8: Final Lock-In Acceptance, Holdout-750, Speed/VRAM/NFE & Fixed Visual Strips
# ---------------------------------------------------------------------------

def _export_fixed_task_visual_grid() -> dict[str, Any]:
    """Assemble fixed task_id visual comparison strips (5 cross-style tasks) into docs/performance/figures/."""
    tasks, _, _ = tasks_and_sources()
    # Pick 1 canonical deterministic cross-style task per target style (tid=0..4, source != target)
    picked = []
    for tid in range(5):
        cand = [t for t in tasks if int(t["target_style_id"]) == tid and int(t["source_style_id"]) != tid]
        picked.append(cand[3])  # deterministic index across styles

    fig_dir = ROOT / "docs/performance/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Columns: Target Ref / Source / E-only(l100) / P-only(s085) / D0_sdvae / D1_wikiart / CLEAVE(s085) / CLEAVE(s100)
    col_specs = [
        ("E-only (l1.0)",    lambda tid_s: OUT / "raw/P7_Eonly_l100"),
        ("P-only (s0.85)",   lambda tid_s: OUT / "raw/P7_Ponly_src_chroma_s085"),
        ("P2 D0 SDVAE",      lambda tid_s: OUT / "raw/P2_pcore_s085_D0_sdvae"),
        ("P2 D1 WikiFT",     lambda tid_s: OUT / "raw/P2_pcore_s085_D1_wikiart_ft"),
        ("CLEAVE s=0.85",    lambda tid_s: OUT / "raw/P8_CLEAVE_Final_s085"),
        ("CLEAVE s=1.00",    lambda tid_s: OUT / "raw/P8_CLEAVE_Final_s100"),
        ("CLEAVE s=1.25",    lambda tid_s: OUT / "raw/P8_CLEAVE_Final_s125"),
    ]
    src_feats = torch.load(ROOT / "runs/v6_r2/cache/tasks750_source_features.pt", map_location="cpu", weights_only=False)

    def _find_img_in_raw(raw_dir: Path, task_id: str) -> Image.Image | None:
        meta_p = raw_dir / "meta.json"
        if not meta_p.exists():
            return None
        meta = read_json(meta_p)
        for b in meta["batches"]:
            if task_id in b["ids"]:
                idx = b["ids"].index(task_id)
                pl = torch.load(raw_dir / b["file"], map_location="cpu", weights_only=False)
                arr = pl["rgb_u8"][idx].permute(1, 2, 0).numpy()
                return Image.fromarray(arr)
        return None

    cell = 224
    pad = 8
    header_h = 28
    n_cols = 1 + len(col_specs)
    n_rows = len(picked)
    W = n_cols * cell + (n_cols + 1) * pad
    H = header_h + n_rows * cell + (n_rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (20, 22, 26))
    draw = ImageDraw.Draw(canvas)

    headers = ["Source"] + [c[0] for c in col_specs]
    for ci, h_txt in enumerate(headers):
        x0 = pad + ci * (cell + pad)
        draw.text((x0 + 4, 6), h_txt, fill=(235, 240, 248))

    for ri, t in enumerate(picked):
        tid_s = t["task_id"]
        y0 = header_h + pad + ri * (cell + pad)
        src_t = src_feats["raw_rgb"][t["source_stem"]]
        src_u8 = (src_t.clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).numpy()
        im_src = Image.fromarray(src_u8).resize((cell, cell), Image.Resampling.BILINEAR)
        canvas.paste(im_src, (pad, y0))
        draw.text((pad + 4, y0 + cell - 16), f"{tid_s} -> {t['target_style'][:6]}", fill=(255, 230, 140))

        for ci, (_, dir_fn) in enumerate(col_specs, start=1):
            x0 = pad + ci * (cell + pad)
            im = _find_img_in_raw(dir_fn(tid_s), tid_s)
            if im is not None:
                canvas.paste(im.resize((cell, cell), Image.Resampling.BILINEAR), (x0, y0))

    grid_p = fig_dir / "v35_fixed_task_comparison.png"
    canvas.save(grid_p, format="PNG")
    return {
        "grid_path": str(grid_p.relative_to(ROOT)),
        "task_ids": [{"task_id": t["task_id"], "src": t["source_style"], "tgt": t["target_style"]} for t in picked],
    }


def run_p8(*, device: str = "cuda") -> dict[str, Any]:
    rep_p = OUT / "p8_final_report.json"
    if rep_p.exists():
        print(f"[Stage P8] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    t0 = time.monotonic()
    p3_rep = read_json(OUT / "p3_stiefel_poolc_report.json")
    p4_rep = read_json(OUT / "p4_seed2_report.json")
    p5_rep = read_json(OUT / "p5_duration_report.json")
    p6_rep = read_json(OUT / "p6_hyper_report.json")
    fcfg = p6_rep["final_schedule_cfg"]

    locked_ckpt = Path(p5_rep["locked_ckpt"])
    locked_wkey = str(p5_rep["locked_weight"])
    locked_step = int(p5_rep["locked_step"])

    # 1. Final full test750 spectrum + wrong-target check
    m_final, _ = load_pcore_v35(locked_ckpt, weight_key=locked_wkey, device=device)
    final_levels = eval_pcore_test750(
        "P8_CLEAVE_Final", m_final,
        knob_list=(0.75, 0.85, 1.00, 1.25, 1.50),
        include_tau0=True, eval_wrong=True,
        alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"],
        ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"], steps=int(fcfg["steps"]),
        device=device,
    )
    for k in list(final_levels.keys()):
        final_levels[k].update(_extra_cross_rates(OUT / "measured" / f"P8_CLEAVE_Final_{k}"))

    # 2. If P5 chose longer training (>3500 steps), note or check P4 seed status
    seed2_note = "P4 R0 trained 3500 steps; matched locked step=3500." if locked_step == 3500 else f"P5 locked {locked_step} steps on Seed-1; P4 Seed-2 was 3500 steps (single-seed long run noted per P8 protocol)."

    # 3. Holdout-750 verification on CLEAVE Final vs P4 R0 (Seed-2) vs V32 Base
    ho_final = evaluate_holdout_v35(m_final, tag="P8_CLEAVE_Final_ho_s085", fcfg=fcfg, knob_s=0.85, device=device)
    del m_final
    torch.cuda.empty_cache()

    m_r0, _ = load_pcore_v35(Path(p4_rep["trains"]["P4_R0_stiefel_seed2"]["ckpt"]), device=device)
    ho_r0 = evaluate_holdout_v35(m_r0, tag="P8_Seed2_R0_ho_s085", fcfg=fcfg, knob_s=0.85, device=device)
    del m_r0
    torch.cuda.empty_cache()

    # 4. Cold-start RTX 4090 RGB->RGB latency + VRAM + NFE breakdown
    perf = measure_e2e_speed_and_params(locked_ckpt, fcfg, weight_key=locked_wkey, device=device)

    # 5. Export fixed-task_id visual strip
    vis_strip = _export_fixed_task_visual_grid()

    # 6. Paired CI vs V32 on test750 if available
    v32_s085 = OUT.parent / "v32_nodino/measured/PCore_NoDINO_s085_c10/eval.json"
    ci_vs_v32 = {}
    if v32_s085.exists():
        ci_vs_v32["s085_vs_V32"] = paired_ci_vs_baseline(OUT / "measured/P8_CLEAVE_Final_s085/eval.json", v32_s085)

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P8_final_acceptance",
        "locked_ckpt": str(locked_ckpt),
        "locked_weight": locked_wkey,
        "locked_step": locked_step,
        "final_schedule_cfg": fcfg,
        "seed2_long_note": seed2_note,
        "test750_matched_c": summarize_matched_c(final_levels),
        "test750_levels": final_levels,
        "holdout_s085": {"final": ho_final, "seed2_r0": ho_r0},
        "perf_4090": perf,
        "ci_vs_v32": ci_vs_v32,
        "visual_strip": vis_strip,
        "p3_matched_c_summary": {k: v["matched_c"] for k, v in p3_rep["arms"].items()},
        "p4_matched_c_summary": {k: v["matched_c"] for k, v in p4_rep["arms"].items()},
    }
    save_json(rep_p, rep)
    print(f"[Stage P8 DONE] test750 S@50,61,70={rep['test750_matched_c']['dino_s_at_c']} | b16={perf['speed']['b16_ips']} img/s | b1={perf['speed']['b1_ms_per_img']}ms", flush=True)
    return rep
