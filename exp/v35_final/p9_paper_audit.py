"""CLEAVE V35 Stage P9: Surgical Paper Finalization Audits & Controls.

Closes every open quantitative/methodological check before paper text freeze:
  - P9a (P0 + P1): Real high-content P-Core measurements (s=0.60..0.72, eliminating C=0.70 boundary
    clamping), unified E-only (A0 vs A3, lam=0.0..1.0), and dense E/P boundary continuity audit.
  - P9b (P0 + P1): Unified Decoder evaluation (D0/D1/D2) strictly on P8 calibrated latents
    (s085, s100, s150, ecore_a3_l090) + wrong-target directional specificity (s085_wrong, s100_wrong).
  - P9c (P0): Unified E/P system comparison under identical fcfg (E-only A0/A3 vs P-only raw/A1/A3 vs CLEAVE).
  - P9d (P0): Corrected real-artist Holdout-750 & test750 overlap audit, s=1.00 Holdout-750 (Seed-1 & Seed-2),
    and unseen-artist vs work-disjoint slicing.
  - P9e (P2): Iso-parameter 58.04M Shared Single-Backbone (Unified58M, base_dim=240) training &
    evaluation (self-draft 1+6 NFE, src_chroma, tau0=0) + 4090 cold-start speed vs decoupled E+P (55.67M).

Zero-DINO AST hygiene strictly enforced.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v34_frontier import enrich_metrics_v34
from exp.v35_final import (
    D0_FULL_VAE_DIR,
    D1_FULL_FT_CKPT,
    OUT,
    S10B_CKPT,
    RawWriterV35,
    build_pcore_startpoint,
    measure_and_summarize_batch_v35,
    paired_ci_vs_baseline,
    summarize_matched_c,
)
from exp.v35_final.common import (
    eval_pcore_test750,
    evaluate_holdout_v35,
    load_pcore_v35,
    load_v32_ecore_and_priors,
    make_v35_schedule,
    measure_e2e_speed_and_params,
    train_44m_ablation,
)
from exp.v35_final.p0_p2_evals import (
    _artist_key,
    _extra_cross_rates,
    _load_autoencoder_decoder,
    _measure_texture_stats,
)
from src.eval.holdout import build_or_load_holdout_spec
from src.eval.metrics import paired_cluster_ci, summarize_rows
from src.utils import auto_decode_batch, auto_infer_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, canonical_pid, pick_wrong_cross_tid
from src.utils.runtime import read_json, sanitize_cuda_env, save_json, utc


def _load_p8_cfg() -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    p8 = read_json(OUT / "p8_final_report.json")
    return (
        Path(p8["locked_ckpt"]),
        str(p8["locked_weight"]),
        dict(p8["final_schedule_cfg"]),
        p8,
    )


def _eval_ecore_sweep(
    prefix: str,
    lam_list: tuple[float, ...],
    *,
    use_a3: bool,
    device: str = "cuda",
) -> dict[str, dict[str, Any]]:
    tasks, z_src, _ = tasks_and_sources()
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    assets = build_or_load_cnx_assets(device=device)
    q16_750 = assets["stiefel"]["test750"]["q32"].float()
    c64_750 = ts["c_64"].float()
    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    dec = load_s10_decoder(device)
    infer_bs = auto_infer_batch(device)

    arms: list[tuple[str, str]] = []
    for lam in lam_list:
        ltag = f"l{int(round(lam * 100)):03d}"
        arm = f"{prefix}_{ltag}"
        arms.append((ltag, arm))
        if not (OUT / "measured" / arm / "summary_v34.json").exists():
            writer = RawWriterV35(arm, dec=dec, device=device)
            for st in range(0, len(tasks), infer_bs):
                sl = slice(st, min(st + infer_bs, len(tasks)))
                ti = tid_750[sl].to(device)
                ze = ecore(
                    z_src[sl].to(device).float(),
                    c64_750[sl].to(device).float(),
                    q16_750[sl].to(device).float(),
                    ti,
                    torch.full((len(ti),), float(lam), device=device),
                    raw_stats,
                    priors_ec,
                    noise=eps_750[sl].to(device).float(),
                )
                if use_a3 and lam > 0.0:
                    ze = build_pcore_startpoint(ze, ti, priors_a, alpha_y=1.0, alpha_c=1.0)
                writer.write_batch(ze, tasks[sl])
            writer.finalize(arm=arm, system=("E+A3" if use_a3 else "E-only"), draft_lambda=lam)

    del ecore, dec
    gc.collect()
    torch.cuda.empty_cache()
    res = measure_and_summarize_batch_v35([a for _, a in arms], device=device)
    out: dict[str, dict[str, Any]] = {}
    for ltag, arm in arms:
        res[arm].update(_extra_cross_rates(OUT / "measured" / arm))
        out[ltag] = res[arm]
    return out


# ---------------------------------------------------------------------------
# P9a: Real High-Content Points + Dense E/P Boundary Continuity
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_p9a(device: str = "cuda") -> dict[str, Any]:
    ckpt_p, wkey, fcfg, p8 = _load_p8_cfg()
    pcore, _ = load_pcore_v35(ckpt_p, weight_key=wkey, device=device)

    # 1. Measure real high-content points on locked P8 Final (brackets C=0.70 and C=0.75!)
    hi_knobs = (0.60, 0.65, 0.68, 0.70, 0.72)
    hi_levels = eval_pcore_test750(
        "P8_CLEAVE_Final", pcore,
        knob_list=hi_knobs,
        eval_wrong=False,
        alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"],
        ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"], steps=int(fcfg["steps"]),
        device=device,
    )
    for k in list(hi_levels.keys()):
        hi_levels[k].update(_extra_cross_rates(OUT / "measured" / f"P8_CLEAVE_Final_{k}"))

    # Combine all 10 measured P8 Final P-Core levels (0.60 .. 1.50)
    full_pcore_levels = {**hi_levels, **{k: v for k, v in p8["test750_levels"].items() if k != "s085_wrong" and k != "tau0"}}
    matched_c_bracketed = summarize_matched_c(full_pcore_levels, c_grid=(0.50, 0.61, 0.70, 0.75))

    # 2. Unified E-only (A0) and E-only (+A3) sweeps across lam=0.00..1.00
    e_lams = (0.00, 0.25, 0.40, 0.50, 0.55, 0.60, 0.65, 0.75, 0.90, 1.00)
    ecore_a0 = _eval_ecore_sweep("P9_E_A0", e_lams, use_a3=False, device=device)
    ecore_a3 = _eval_ecore_sweep("P9_E_A3", e_lams, use_a3=True, device=device)

    # 3. Dense latent trajectory continuity check across E-Core -> P-Core handoff (first 32 cross tasks)
    tasks, z_src, _ = tasks_and_sources()
    sub_idx = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:32]
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    assets = build_or_load_cnx_assets(device=device)
    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)

    zb = z_src[sub_idx].to(device).float()
    c64 = ts["c_64"][sub_idx].to(device).float()
    q16 = assets["stiefel"]["test750"]["q32"][sub_idx].to(device).float()
    cg = assets["stiefel"]["test750"]["cnx_gray"][sub_idx].to(device).float()
    y16 = project_latent_y16_lcn(zb, v18["luma_proj"]["w"].to(device).float(), v18["luma_proj"]["b"].to(device).float())
    ti = torch.tensor([tasks[i]["target_style_id"] for i in sub_idx], device=device, dtype=torch.long)
    u_st = match_cnx_k8_submode(cg, ti, assets["bank"])
    eps = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))[sub_idx].to(device).float()

    dense_pts: list[tuple[str, torch.Tensor]] = []
    for lam in (0.50, 0.55, 0.60):
        ze = ecore(zb, c64, q16, ti, torch.full((len(ti),), lam, device=device), raw_stats, priors_ec, noise=eps)
        ze_a3 = build_pcore_startpoint(ze, ti, priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
        dense_pts.append((f"E+A3_l{int(round(lam*100)):03d}", ze_a3))
    for sv in (0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.85):
        kw = make_v35_schedule(sv, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
        lam_s = float(kw.pop("draft_lambda", 0.75))
        ze = ecore(zb, c64, q16, ti, torch.full((len(ti),), lam_s, device=device), raw_stats, priors_ec, noise=eps)
        ze_s = build_pcore_startpoint(ze, ti, priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
        zo = integrate_v33_trajectory(pcore, ze_s, eps, y16, q16, ti, u_st, steps=int(fcfg["steps"]), solver="dpm2m", **kw)
        dense_pts.append((f"P_s{int(round(sv*100)):03d}", zo))

    handoff_l2: list[dict[str, Any]] = []
    for (n0, z0), (n1, z1) in zip(dense_pts, dense_pts[1:]):
        rmse = float((z1 - z0).pow(2).mean().sqrt().item())
        handoff_l2.append({"from": n0, "to": n1, "latent_rmse": round(rmse, 4)})

    del pcore, ecore
    torch.cuda.empty_cache()
    return {
        "pcore_full_levels": full_pcore_levels,
        "matched_c_bracketed": matched_c_bracketed,
        "ecore_a0_sweep": ecore_a0,
        "ecore_a3_sweep": ecore_a3,
        "handoff_latent_continuity": handoff_l2,
    }


# ---------------------------------------------------------------------------
# P9b: Unified Decoder Evaluation on P8 Calibrated Latents + Directional Audit
# ---------------------------------------------------------------------------

def _wrong_follow_metrics(m_dir: Path, alt_wrong_tid: torch.Tensor) -> dict[str, Any]:
    ev = read_json(m_dir / "eval.json")
    rows = ev["rows"]
    cross_pairs = [(i, r) for i, r in enumerate(rows) if r["is_cross"]]
    n = max(1, len(cross_pairs))
    w_follow = sum(int(int(r["pred_class"]) == int(alt_wrong_tid[i].item())) for i, r in cross_pairs) / n
    src_lock = sum(int(int(r["pred_class"]) == int(r["src_style_id"])) for _, r in cross_pairs) / n
    real_tgt_hit = sum(int(r["target_top1"]) for _, r in cross_pairs) / n
    # Margin wrt wrong target class on wrong-target latent
    w_margins = []
    for i, r in cross_pairs:
        wt = int(alt_wrong_tid[i].item())
        cs = r["csd_scores_5"]
        other_max = max(cs[j] for j in range(5) if j != wt)
        w_margins.append(float(cs[wt]) - float(other_max))
    return {
        "wrong_tgt_follow": round(w_follow, 4),
        "pred_eq_src": round(src_lock, 4),
        "orig_tgt_top1": round(real_tgt_hit, 4),
        "wrong_tgt_margin": round(float(np.mean(w_margins)), 5),
    }


@torch.no_grad()
def run_p9b(device: str = "cuda") -> dict[str, Any]:
    ckpt_p, wkey, fcfg, _ = _load_p8_cfg()
    tasks, z_src, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    tid_wrong = pick_wrong_cross_tid(sid_750, tid_750)
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    assets = build_or_load_cnx_assets(device=device)
    q16_750 = assets["stiefel"]["test750"]["q32"].float()
    cnx_gray_750 = assets["stiefel"]["test750"]["cnx_gray"].float()
    y16_750 = project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    c64_750 = ts["c_64"].float()

    lat_dir = OUT / "p9/cached_p8_latents"
    lat_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("p8_ecore_a3_l090", "ecore_a3", 0.90, False),
        ("p8_s085",          "pcore",    0.85, False),
        ("p8_s100",          "pcore",    1.00, False),
        ("p8_s150",          "pcore",    1.50, False),
        ("p8_s085_wrong",    "pcore",    0.85, True),
        ("p8_s100_wrong",    "pcore",    1.00, True),
    ]
    lat_paths = {k: lat_dir / f"{k}.pt" for k, _, _, _ in specs}

    if not all(p.exists() for p in lat_paths.values()):
        ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
        pcore, _ = load_pcore_v35(ckpt_p, weight_key=wkey, device=device)
        infer_bs = auto_infer_batch(device)
        ids = [t["task_id"] for t in tasks]

        def _gen_draft(lam: float, is_wrong: bool) -> torch.Tensor:
            chunks = []
            tl = tid_wrong if is_wrong else tid_750
            for st in range(0, len(tasks), infer_bs):
                sl = slice(st, min(st + infer_bs, len(tasks)))
                ti = tl[sl].to(device)
                ze = ecore(
                    z_src[sl].to(device).float(), c64_750[sl].to(device).float(),
                    q16_750[sl].to(device).float(), ti,
                    torch.full((len(ti),), float(lam), device=device),
                    raw_stats, priors_ec, noise=eps_750[sl].to(device).float(),
                )
                ze_s = build_pcore_startpoint(ze, ti, priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
                chunks.append(ze_s.cpu().half())
            return torch.cat(chunks, 0)

        for key, kind, val, is_wrong in specs:
            if lat_paths[key].exists():
                continue
            if kind == "ecore_a3":
                zt = _gen_draft(val, is_wrong)
            else:
                kw = make_v35_schedule(val, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
                lam = float(kw.pop("draft_lambda", 0.75))
                ze_all = _gen_draft(lam, is_wrong)
                tl = tid_wrong if is_wrong else tid_750
                chunks = []
                for st in range(0, len(tasks), infer_bs):
                    sl = slice(st, min(st + infer_bs, len(tasks)))
                    ti = tl[sl].to(device)
                    u_st = match_cnx_k8_submode(cnx_gray_750[sl].to(device).float(), ti, assets["bank"])
                    zo = integrate_v33_trajectory(
                        pcore, ze_all[sl].to(device).float(), eps_750[sl].to(device).float(),
                        y16_750[sl].to(device).float(), q16_750[sl].to(device).float(),
                        ti, u_st, steps=int(fcfg["steps"]), solver="dpm2m", **kw,
                    )
                    chunks.append(zo.cpu().half())
                zt = torch.cat(chunks, 0)
            torch.save({"ids": ids, "z": zt}, lat_paths[key])
        del ecore, pcore
        gc.collect()
        torch.cuda.empty_cache()

    cnx = load_convnext_tiny_offline(device)
    cnx_s1 = cnx[:2].eval().requires_grad_(False)
    cnx_s2 = cnx[2:4].eval().requires_grad_(False)
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    dec_specs = [
        ("D0_sdvae", "full", None),
        ("D1_wikiart_ft", "full", D1_FULL_FT_CKPT),
        ("D2_s10b", "s10", S10B_CKPT),
    ]
    tex_p = OUT / "p9/decoder_texture_stats.json"
    saved_tex = read_json(tex_p) if tex_p.exists() else {}
    new_tex: dict[str, Any] = {}
    arms_to_meas: list[str] = []

    for d_tag, kind, ckpt_d in dec_specs:
        if kind == "full":
            dec_obj = _load_autoencoder_decoder(ckpt_d, device=device)
            def _dec(z: torch.Tensor, _dec_o=dec_obj) -> torch.Tensor:
                outs = []
                for i in range(0, len(z), 16):
                    o = _dec_o.decode(z[i:i + 16].float().to(device) / 0.18215).sample
                    outs.append((o / 2.0 + 0.5).clamp(0.0, 1.0))
                return torch.cat(outs, 0)
        else:
            dec_obj = load_s10_decoder(device)
            bs_d = auto_decode_batch(device)
            def _dec(z: torch.Tensor, _dec_o=dec_obj, _bs=bs_d) -> torch.Tensor:
                return decode_s10(_dec_o, z.to(device).float(), batch=_bs, output_cpu=False).clamp(0.0, 1.0)

        for l_tag, lp in lat_paths.items():
            arm = f"P9B_{l_tag}_{d_tag}"
            arms_to_meas.append(arm)
            if not (OUT / "measured" / arm / "summary_v34.json").exists():
                z_all = torch.load(lp, map_location="cpu", weights_only=False)["z"]
                writer = RawWriterV35(arm, dec=None, device=device)
                accum_t: dict[str, float] = {}
                n_ch = 0
                for st in range(0, len(tasks), 64):
                    sl = slice(st, min(st + 64, len(tasks)))
                    rgb = _dec(z_all[sl])
                    writer.write_batch(z_all[sl], tasks[sl], rgb_override=rgb)
                    ts_b = _measure_texture_stats(rgb, cnx_s1, cnx_s2, im_mean, im_std)
                    for tk, tv in ts_b.items():
                        accum_t[tk] = accum_t.get(tk, 0.0) + tv
                    n_ch += 1
                new_tex[arm] = {tk: round(tv / max(1, n_ch), 6) for tk, tv in accum_t.items()}
                writer.finalize(arm=arm, decoder=d_tag, latent_source=l_tag)
        del dec_obj
        gc.collect()
        torch.cuda.empty_cache()

    del cnx, cnx_s1, cnx_s2
    torch.cuda.empty_cache()
    if new_tex:
        saved_tex.update(new_tex)
        save_json(tex_p, saved_tex)

    measured = measure_and_summarize_batch_v35(arms_to_meas, device=device)
    table: dict[str, dict[str, Any]] = {}
    paired_cis: dict[str, Any] = {}
    directional: dict[str, dict[str, Any]] = {}

    for l_tag in lat_paths:
        table[l_tag] = {}
        for d_tag, _, _ in dec_specs:
            arm = f"P9B_{l_tag}_{d_tag}"
            row = dict(measured[arm])
            m_dir = OUT / "measured" / arm
            row["csd_margin"] = round(float(read_json(m_dir / "summary_v34.json")["cross600"]["csd_margin"]), 5)
            if "wrong" in l_tag:
                row.update(_wrong_follow_metrics(m_dir, tid_wrong))
            else:
                row.update(_extra_cross_rates(m_dir))
            if arm in saved_tex:
                row["texture"] = saved_tex[arm]
            table[l_tag][d_tag] = row

        if "wrong" not in l_tag:
            r_d2 = read_json(OUT / f"measured/P9B_{l_tag}_D2_s10b/eval.json")["rows"]
            for bd in ("D0_sdvae", "D1_wikiart_ft"):
                paired_cis[f"{l_tag}_D2_minus_{bd}"] = paired_cluster_ci(
                    r_d2, read_json(OUT / f"measured/P9B_{l_tag}_{bd}/eval.json")["rows"]
                )

    for s_tag in ("s085", "s100"):
        directional[s_tag] = {}
        for d_tag, _, _ in dec_specs:
            mc = table[f"p8_{s_tag}"][d_tag]
            mw = table[f"p8_{s_tag}_wrong"][d_tag]
            directional[s_tag][d_tag] = {
                "correct_top1": mc["csd_top1"],
                "wrong_run_tgt_follow": mw["wrong_tgt_follow"],
                "net_directional_lift": round(mc["csd_top1"] - mw["wrong_tgt_follow"], 4),
                "correct_margin": mc["csd_margin"],
                "wrong_run_tgt_margin": mw["wrong_tgt_margin"],
                "margin_spread": round(mc["csd_margin"] - mw["wrong_tgt_margin"], 5),
                "correct_dino_s": mc["dino_s"],
                "wrong_dino_s": mw["dino_s"],
                "dino_s_spread": round(mc["dino_s"] - mw["dino_s"], 4),
            }

    return {"table": table, "paired_ci": paired_cis, "directional_response": directional}


# ---------------------------------------------------------------------------
# P9c: Unified E/P Necessity Across Identical fcfg & Startpoint Variants
# ---------------------------------------------------------------------------

def run_p9c(device: str = "cuda") -> dict[str, Any]:
    ckpt_p, wkey, fcfg, _ = _load_p8_cfg()
    pcore, _ = load_pcore_v35(ckpt_p, weight_key=wkey, device=device)
    knobs = (0.65, 0.75, 0.85, 1.00, 1.25, 1.50)

    variants = [
        ("P9_Ponly_raw_src",    "raw_src"),
        ("P9_Ponly_chroma_a1",  "raw_src_chroma"),
        ("P9_Ponly_full_a3",    "raw_src_a3"),
    ]
    out: dict[str, Any] = {}
    for arm_name, mode in variants:
        lv = eval_pcore_test750(
            arm_name, pcore,
            knob_list=knobs, eval_wrong=False,
            alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"],
            ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"], steps=int(fcfg["steps"]),
            startpoint_mode=mode,
            device=device,
        )
        for k in list(lv.keys()):
            lv[k].update(_extra_cross_rates(OUT / "measured" / f"{arm_name}_{k}"))
        out[arm_name] = {"levels": lv, "matched_c": summarize_matched_c(lv)}

    del pcore
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# P9d: Corrected Real-Artist Audit & Work-Disjoint Holdout (s=0.85 & s=1.00)
# ---------------------------------------------------------------------------

def _slice_eval_by_artist(eval_json_p: Path, tr_art: set[str], styles: list[str]) -> dict[str, Any]:
    ev = read_json(eval_json_p)
    rows = ev["rows"]
    seen_rows, unseen_rows = [], []
    for r in rows:
        art = _artist_key(r["src_stem"])
        (unseen_rows if art not in tr_art else seen_rows).append(r)

    def _agg(sub: list[dict[str, Any]]) -> dict[str, Any]:
        cross = [x for x in sub if x["is_cross"]]
        same = [x for x in sub if not x["is_cross"]]
        if not cross:
            return {"n_cross": 0}
        return {
            "n_cross": len(cross),
            "n_same": len(same),
            "dino_c": round(sum(x["dino_c"] for x in cross) / len(cross), 4),
            "dino_s": round(sum(x["dino_s"] for x in cross) / len(cross), 4),
            "csd_top1": round(sum(x["target_top1"] for x in cross) / len(cross), 4),
            "lpips": round(sum(x["lpips"] for x in cross) / len(cross), 4),
            "same_top1": round(sum(x["target_top1"] for x in same) / max(1, len(same)), 4),
        }

    return {
        "work_disjoint_all": summarize_rows(rows, styles=styles, require_full=True)["cross600"],
        "artist_seen_src": _agg(seen_rows),
        "artist_unseen_src": _agg(unseen_rows),
    }


def run_p9d(device: str = "cuda") -> dict[str, Any]:
    ckpt_p, wkey, fcfg, _ = _load_p8_cfg()
    p4_rep = read_json(OUT / "p4_seed2_report.json")
    r0_ckpt = Path(p4_rep["trains"]["P4_R0_stiefel_seed2"]["ckpt"])

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr_ids = [str(x) for x in v18["train"]["ids"]]
    ho_ids = [str(x) for x in v18["holdout"]["ids"]]
    tr_art = {_artist_key(x) for x in tr_ids}
    ho_spec = build_or_load_holdout_spec(device=device)
    tasks750, _, _ = tasks_and_sources()

    src_idx = list(ho_spec["src_idx"].tolist())
    ref_idx = list(ho_spec["ref_idx"].tolist())
    ho_src_stems = [ho_ids[i] for i in src_idx]
    ho_ref_stems = [ho_ids[i] for i in ref_idx]
    ho_src_unseen = [i for i, s in zip(src_idx, ho_src_stems) if _artist_key(s) not in tr_art]
    ho_ref_unseen = [i for i, s in zip(ref_idx, ho_ref_stems) if _artist_key(s) not in tr_art]
    ts_src_stems = [t["source_stem"] for t in tasks750[::5]]
    ts_src_unseen = [s for s in ts_src_stems if _artist_key(s) not in tr_art]

    artist_audit = {
        "train_4400_unique_artists": len(tr_art),
        "holdout_pid_overlap_train": len({canonical_pid(x) for x in ho_ids} & {canonical_pid(x) for x in tr_ids}),
        "holdout_pid_overlap_test750": len({canonical_pid(x) for x in ho_ids} & {canonical_pid(t["source_stem"]) for t in tasks750}),
        "holdout_src_seen_artists_in_train": f"{150 - len(ho_src_unseen)}/150",
        "holdout_src_unseen_artist_items": len(ho_src_unseen),
        "holdout_ref_seen_artists_in_train": f"{150 - len(ho_ref_unseen)}/150",
        "holdout_ref_unseen_artist_items": len(ho_ref_unseen),
        "test750_src_unseen_artist_items": len(ts_src_unseen),
        "classification": "Work-Disjoint (0 painting overlap) Validation — NOT Artist-Disjoint OOD (WikiArt train_4400 already includes 372 artists covering 146/150 holdout sources).",
    }

    m_f, _ = load_pcore_v35(ckpt_p, weight_key=wkey, device=device)
    ho_f_085 = evaluate_holdout_v35(m_f, tag="P8_CLEAVE_Final_ho_s085", fcfg=fcfg, knob_s=0.85, device=device)
    ho_f_100 = evaluate_holdout_v35(m_f, tag="P9_CLEAVE_Final_ho_s100", fcfg=fcfg, knob_s=1.00, device=device)
    del m_f
    torch.cuda.empty_cache()

    m_r, _ = load_pcore_v35(r0_ckpt, weight_key="ema", device=device)
    ho_r_085 = evaluate_holdout_v35(m_r, tag="P8_Seed2_R0_ho_s085", fcfg=fcfg, knob_s=0.85, device=device)
    ho_r_100 = evaluate_holdout_v35(m_r, tag="P9_Seed2_R0_ho_s100", fcfg=fcfg, knob_s=1.00, device=device)
    del m_r
    torch.cuda.empty_cache()

    styles_list = list(read_json(OUT / "measured/P8_CLEAVE_Final_s085/summary_v34.json")["per_target_cross120"].keys())
    slices = {
        "holdout_final_s085": _slice_eval_by_artist(OUT / "holdout/measured/P8_CLEAVE_Final_ho_s085/eval.json", tr_art, styles_list),
        "holdout_final_s100": _slice_eval_by_artist(OUT / "holdout/measured/P9_CLEAVE_Final_ho_s100/eval.json", tr_art, styles_list),
        "test750_final_s085": _slice_eval_by_artist(OUT / "measured/P8_CLEAVE_Final_s085/eval.json", tr_art, styles_list),
    }

    return {
        "corrected_artist_audit": artist_audit,
        "holdout_levels": {
            "final_s085": ho_f_085,
            "final_s100": ho_f_100,
            "seed2_s085": ho_r_085,
            "seed2_s100": ho_r_100,
        },
        "artist_slices": slices,
    }


# ---------------------------------------------------------------------------
# P9e: Iso-Parameter Shared Single-Backbone (Unified58M, base_dim=240)
# ---------------------------------------------------------------------------

def run_p9e(device: str = "cuda") -> dict[str, Any]:
    _, _, fcfg, _ = _load_p8_cfg()
    tr_info = train_44m_ablation(
        "P9_Unified58M",
        OUT / "p9/P9_Unified58M",
        base_dim=240,
        cond_dim=384,
        content_mode="stiefel",
        pool_c_mode="15pct",
        updates=3500,
        seed_offset=0,
        device=device,
    )
    u_ckpt = Path(tr_info["ckpt"])
    u_model, _ = load_pcore_v35(u_ckpt, weight_key="ema", device=device)

    shared_drafts: dict[tuple[float, bool], torch.Tensor] = {}
    lv_self_ep = eval_pcore_test750(
        "P9_U58_self_ep", u_model,
        knob_list=(0.65, 0.75, 0.85, 1.00, 1.25, 1.50),
        include_tau0=True, eval_wrong=True,
        alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"],
        ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"], steps=int(fcfg["steps"]),
        startpoint_mode="self_ep",
        shared_drafts=shared_drafts,
        device=device,
    )
    lv_src_chroma = eval_pcore_test750(
        "P9_U58_src_chroma", u_model,
        knob_list=(0.75, 0.85, 1.00, 1.25),
        include_tau0=False, eval_wrong=False,
        alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"],
        ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"], steps=int(fcfg["steps"]),
        startpoint_mode="raw_src_chroma",
        shared_drafts=shared_drafts,
        device=device,
    )
    del u_model
    torch.cuda.empty_cache()

    for prefix, lv in (("P9_U58_self_ep", lv_self_ep), ("P9_U58_src_chroma", lv_src_chroma)):
        for k in list(lv.keys()):
            lv[k].update(_extra_cross_rates(OUT / "measured" / f"{prefix}_{k}"))

    perf_u58 = measure_e2e_speed_and_params(u_ckpt, fcfg, weight_key="ema", device=device)
    cis: dict[str, Any] = {}
    for lvl in ("s085", "s100", "s125", "s150", "tau0"):
        p_ep = OUT / f"measured/P8_CLEAVE_Final_{lvl}/eval.json"
        p_u58 = OUT / f"measured/P9_U58_self_ep_{lvl}/eval.json"
        if p_ep.exists() and p_u58.exists():
            cis[f"CLEAVE_EP_minus_Unified58M_selfEP_{lvl}"] = paired_ci_vs_baseline(p_ep, p_u58)

    return {
        "train_info": tr_info,
        "param_comparison": {
            "cleave_ep_flow_params": 11_131_268 + 44_540_532,
            "unified58m_flow_params": int(tr_info["cfg"]["pcore_params"]),
        },
        "self_ep_matched_c": summarize_matched_c(lv_self_ep, c_grid=(0.50, 0.61, 0.70, 0.75)),
        "self_ep_levels": lv_self_ep,
        "src_chroma_matched_c": summarize_matched_c(lv_src_chroma, c_grid=(0.50, 0.61, 0.70)),
        "src_chroma_levels": lv_src_chroma,
        "perf_4090_u58": perf_u58,
        "paired_ci_vs_cleave_ep": cis,
    }


def run_p9(*, device: str = "cuda", only: str | None = None) -> dict[str, Any]:
    sanitize_cuda_env()
    rep_p = OUT / "p9_paper_audit_report.json"
    rep = read_json(rep_p) if rep_p.exists() else {"stage": "P9_paper_audit"}
    t0 = time.monotonic()

    tasks_map = [
        ("p9a", "p9a_high_content_and_handoff", run_p9a),
        ("p9b", "p9b_unified_decoder_directional", run_p9b),
        ("p9c", "p9c_unified_ep_necessity", run_p9c),
        ("p9d", "p9d_artist_and_work_holdout", run_p9d),
        ("p9e", "p9e_iso_param_shared_backbone", run_p9e),
    ]
    for short_k, full_k, fn in tasks_map:
        if only is not None and only != short_k:
            continue
        if full_k in rep and only is None:
            print(f"[Stage {short_k.upper()}] Reusing cached section in {rep_p}", flush=True)
            continue
        print(f"=== Running {full_k} ===", flush=True)
        rep[full_k] = fn(device=device)
        rep["updated_utc"] = utc()
        rep["wall_sec"] = round(time.monotonic() - t0, 2)
        save_json(rep_p, rep)

    print(f"[Stage P9 DONE] saved {rep_p}", flush=True)
    return rep
