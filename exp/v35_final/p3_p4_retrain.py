"""CLEAVE V35 Plan Stages P3 (Stiefel vs Whiten-32 vs Raw PCA-32 + Pool C Ablation + Probes + 6-Arm Diag)
and P4 (Second-Seed Verification of Stiefel vs Plain Whitening).

Zero-DINO AST hygiene strictly maintained.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from exp.v18.structure import _train_linear_mlp_probe
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import _gram_feat
from exp.v35_final import C2_V34_CKPT, OUT, paired_ci_vs_baseline, summarize_matched_c
from exp.v35_final.common import (
    build_or_load_content_ablation_reps,
    eval_pcore_test750,
    load_pcore_v35,
    train_44m_ablation,
)
from exp.v35_final.p0_p2_evals import _extra_cross_rates
from src.utils.runtime import read_json, save_json, utc


def _spatial_feat(t4d: torch.Tensor) -> torch.Tensor:
    """Flattened 4x4 spatially pooled feature vector (B, 32*4*4=512) to audit spatial style leakage."""
    return F.avg_pool2d(t4d.float(), 4).flatten(1)


def run_representation_probes(device: str = "cuda") -> dict[str, Any]:
    """Run 5-way train_4400 -> holdout_300 linear/MLP Gram Probe & Spatial Probe across Q0/Q1/Q2 + raw cnx_gray."""
    reps = build_or_load_content_ablation_reps(device=device)
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    sid_tr = v18["train"]["sid"].long()
    sid_ho = v18["holdout"]["sid"].long()

    assets_st = torch.load(OUT.parent / "v32_nodino/cache/cnx_stiefel_cache.pt", map_location="cpu", weights_only=False)
    tr_gray = assets_st["train"]["cnx_gray"]

    probes: dict[str, dict[str, Any]] = {}
    with torch.enable_grad():
        for name in ("stiefel", "whiten32", "pca32"):
            x_tr = reps["train"][name]
            x_ho = reps["holdout"][name]
            g_acc = _train_linear_mlp_probe(_gram_feat(x_tr), sid_tr, _gram_feat(x_ho), sid_ho, epochs=120)
            s_acc = _train_linear_mlp_probe(_spatial_feat(x_tr), sid_tr, _spatial_feat(x_ho), sid_ho, epochs=120)
            g_mat = torch.bmm(x_ho.float().flatten(2), x_ho.float().flatten(2).transpose(1, 2)) / 256.0
            off_diag = (g_mat - torch.diag_embed(g_mat.diagonal(dim1=1, dim2=2))).abs().mean().item()
            diag_err = (g_mat.diagonal(dim1=1, dim2=2) - 1.0).abs().mean().item()
            probes[name] = {
                "gram_probe_5way_acc": round(float(g_acc), 4),
                "spatial_probe_5way_acc": round(float(s_acc), 4),
                "gram_offdiag_abs_mean": round(float(off_diag), 4),
                "gram_diag_err": round(float(diag_err), 4),
            }
        # Reference Gram leakage on raw 384-ch cnx_gray
        g_raw = float(assets_st["gram_leak_audit"]["old_cnx16_gram_5way_leak"])
        probes["cnx_gray_raw_384"] = {"gram_probe_5way_acc": round(g_raw, 4)}
    return probes


def run_p3(*, device: str = "cuda") -> dict[str, Any]:
    rep_p = OUT / "p3_stiefel_poolc_report.json"
    if rep_p.exists():
        print(f"[Stage P3] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    t0 = time.monotonic()
    reps = build_or_load_content_ablation_reps(device=device)
    probes = run_representation_probes(device=device)
    print(f"[Stage P3] Probes: {probes}", flush=True)

    # Verify Q0 baseline == V34 C2_44m_q32 exact configuration
    sv_c2 = torch.load(C2_V34_CKPT, map_location="cpu", weights_only=False)
    c2_cfg = sv_c2["cfg"]
    assert int(c2_cfg["base_dim"]) == 208 and int(c2_cfg["cond_dim"]) == 384 and int(c2_cfg["updates"]) == 3500
    assert not bool(c2_cfg.get("use_adapter", False)) and c2_cfg.get("style_mode", "k8") == "k8"

    # Train Q1 (Plain channel-wise whitening), Q2 (Raw PCA-32), Q3 (0% Pool C self-matched)
    arm_defs = [
        ("P3_Q0_stiefel_poolc", "stiefel",  "15pct",           C2_V34_CKPT, None),
        ("P3_Q1_whiten_poolc",  "whiten32", "15pct",           None, {"content_mode": "whiten32", "pool_c_mode": "15pct", "seed_offset": 0}),
        ("P3_Q2_pca32_poolc",   "pca32",    "15pct",           None, {"content_mode": "pca32",    "pool_c_mode": "15pct", "seed_offset": 0}),
        ("P3_Q3_stiefel_nopc",  "stiefel",  "0pct_self_match", None, {"content_mode": "stiefel",  "pool_c_mode": "0pct_self_match", "seed_offset": 0}),
    ]

    arms_eval: dict[str, Any] = {}
    train_infos: dict[str, Any] = {}
    shared_drafts: dict[tuple[float, bool], torch.Tensor] = {}

    for arm_tag, c_mode, pc_mode, existing_ckpt, tr_kw in arm_defs:
        if existing_ckpt is not None:
            ckpt_p = existing_ckpt
            train_infos[arm_tag] = {"reused_v34_c2": str(ckpt_p), "cfg": c2_cfg}
        else:
            assert tr_kw is not None
            tr_info = train_44m_ablation(arm_tag, OUT / "p3" / arm_tag, device=device, **tr_kw)
            ckpt_p = Path(tr_info["ckpt"])
            train_infos[arm_tag] = tr_info

        m, _ = load_pcore_v35(ckpt_p, device=device)
        lv = eval_pcore_test750(
            arm_tag, m,
            knob_list=(0.75, 0.85, 1.00, 1.25, 1.50),
            eval_wrong=True,
            alpha_y=1.0, alpha_c=1.0,
            q32_override=reps["test750"][c_mode],
            shared_drafts=shared_drafts,
            device=device,
        )
        del m
        torch.cuda.empty_cache()

        for k in list(lv.keys()):
            lv[k].update(_extra_cross_rates(OUT / "measured" / f"{arm_tag}_{k}"))
        arms_eval[arm_tag] = {
            "content_mode": c_mode,
            "pool_c_mode": pc_mode,
            "matched_c": summarize_matched_c(lv),
            "levels": lv,
        }

    # Paired Bootstrap CIs: Q0 - Q1 (Stiefel ortho effect), Q0 - Q2 (vs Raw PCA), Q0 - Q3 (Pool C effect)
    cis: dict[str, Any] = {}
    for lvl in ("s085", "s100", "s125"):
        p_q0 = OUT / f"measured/P3_Q0_stiefel_poolc_{lvl}/eval.json"
        for other in ("P3_Q1_whiten_poolc", "P3_Q2_pca32_poolc", "P3_Q3_stiefel_nopc"):
            cis[f"Q0_minus_{other}_{lvl}"] = paired_ci_vs_baseline(p_q0, OUT / f"measured/{other}_{lvl}/eval.json")

    # Active 44M Q0 6-arm Q32 causal intervention: (correct / zero / shuffled) x (s085 [tau0=0.79] / tau0=0.0)
    m_q0, _ = load_pcore_v35(C2_V34_CKPT, device=device)
    diag_6arm: dict[str, Any] = {}
    for start_tag, use_tau0 in (("s085", False), ("tau0", True)):
        for q_m in ("correct", "zero", "shuffled"):
            d_arm = f"P3_diag44m_{start_tag}_{q_m}"
            if use_tau0:
                dlv = eval_pcore_test750(
                    d_arm, m_q0,
                    knob_list=(), include_tau0=True, eval_wrong=False,
                    q_mode=q_m, shared_drafts=shared_drafts, device=device,
                )
                key_out = "tau0"
            else:
                dlv = eval_pcore_test750(
                    d_arm, m_q0,
                    knob_list=(0.85,), include_tau0=False, eval_wrong=False,
                    q_mode=q_m, shared_drafts=shared_drafts, device=device,
                )
                key_out = "s085"
            res = dlv[key_out]
            res.update(_extra_cross_rates(OUT / "measured" / f"{d_arm}_{key_out}"))
            diag_6arm[f"{start_tag}_{q_m}"] = res

    for start_tag in ("s085", "tau0"):
        for bad in ("zero", "shuffled"):
            cis[f"diag44m_{start_tag}_correct_minus_{bad}"] = paired_ci_vs_baseline(
                OUT / f"measured/P3_diag44m_{start_tag}_correct_{start_tag}/eval.json",
                OUT / f"measured/P3_diag44m_{start_tag}_{bad}_{start_tag}/eval.json",
            )

    del m_q0
    torch.cuda.empty_cache()

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P3_stiefel_and_poolc_ablation",
        "probes": probes,
        "train_infos": train_infos,
        "arms": arms_eval,
        "diag_6arm_44m": diag_6arm,
        "paired_ci": cis,
    }
    save_json(rep_p, rep)
    print(f"[Stage P3 DONE] Q0 S@50,61,70={arms_eval['P3_Q0_stiefel_poolc']['matched_c']['dino_s_at_c']} vs Q1={arms_eval['P3_Q1_whiten_poolc']['matched_c']['dino_s_at_c']} vs Q3={arms_eval['P3_Q3_stiefel_nopc']['matched_c']['dino_s_at_c']}", flush=True)
    return rep


def run_p4(*, device: str = "cuda") -> dict[str, Any]:
    """Train & evaluate Seed-2 for R0 (44M + Stiefel + Pool C) and R1 (44M + Whiten32 + Pool C)."""
    rep_p = OUT / "p4_seed2_report.json"
    if rep_p.exists():
        print(f"[Stage P4] Reusing completed {rep_p}", flush=True)
        return read_json(rep_p)

    t0 = time.monotonic()
    reps = build_or_load_content_ablation_reps(device=device)
    shared_drafts: dict[tuple[float, bool], torch.Tensor] = {}

    r_specs = [
        ("P4_R0_stiefel_seed2", "stiefel",  "15pct", 999),
        ("P4_R1_whiten_seed2",  "whiten32", "15pct", 999),
    ]
    arms: dict[str, Any] = {}
    trains: dict[str, Any] = {}

    for arm_tag, c_mode, pc_mode, seed_off in r_specs:
        tr_info = train_44m_ablation(
            arm_tag, OUT / "p4" / arm_tag,
            content_mode=c_mode, pool_c_mode=pc_mode,
            updates=3500, seed_offset=seed_off, device=device,
        )
        trains[arm_tag] = tr_info
        m, _ = load_pcore_v35(Path(tr_info["ckpt"]), device=device)
        lv = eval_pcore_test750(
            arm_tag, m,
            knob_list=(0.75, 0.85, 1.00, 1.25, 1.50),
            eval_wrong=True,
            alpha_y=1.0, alpha_c=1.0,
            q32_override=reps["test750"][c_mode],
            shared_drafts=shared_drafts,
            device=device,
        )
        del m
        torch.cuda.empty_cache()
        for k in list(lv.keys()):
            lv[k].update(_extra_cross_rates(OUT / "measured" / f"{arm_tag}_{k}"))
        arms[arm_tag] = {"content_mode": c_mode, "matched_c": summarize_matched_c(lv), "levels": lv}

    cis: dict[str, Any] = {}
    for lvl in ("s085", "s100", "s125"):
        cis[f"R0_minus_R1_{lvl}"] = paired_ci_vs_baseline(
            OUT / f"measured/P4_R0_stiefel_seed2_{lvl}/eval.json",
            OUT / f"measured/P4_R1_whiten_seed2_{lvl}/eval.json",
        )
        p_q0 = OUT / f"measured/P3_Q0_stiefel_poolc_{lvl}/eval.json"
        if p_q0.exists():
            cis[f"R0_minus_Q0_{lvl}"] = paired_ci_vs_baseline(
                OUT / f"measured/P4_R0_stiefel_seed2_{lvl}/eval.json",
                p_q0,
            )

    rep = {
        "updated_utc": utc(),
        "wall_sec": round(time.monotonic() - t0, 2),
        "stage": "P4_second_seed_verification",
        "trains": trains,
        "arms": arms,
        "paired_ci": cis,
    }
    save_json(rep_p, rep)
    print(f"[Stage P4 DONE] R0 S@50,61,70={arms['P4_R0_stiefel_seed2']['matched_c']['dino_s_at_c']} vs R1={arms['P4_R1_whiten_seed2']['matched_c']['dino_s_at_c']}", flush=True)
    return rep
