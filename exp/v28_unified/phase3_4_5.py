"""Phases 3, 4, 5: Student Architecture Repair (M0-M3), 6-Arm Teacher Duel, and 5k Dual-Seed Finalization."""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from exp.v18 import STYLES, tasks_and_sources
from exp.v19.aniso_teacher import _smooth_grad_3x3, aniso_rms_direction, project_anisotropic_trust
from exp.v19.fiber_duality import UnnormalizedVGGFeatureExtractor
from exp.v27_mainline.followup.reference_ab import band_signature, normalized_gray
from exp.v27_single.bare_q32 import load_cache
from exp.v27_single.unified.train import haar_signature
from exp.v28_unified import (
    OUT,
    ROOT,
    SEED_TRAIN,
    UNIFIED_OLD_OUT,
    RawWriterV28,
    assert_vram_budget,
    run_measure_v28,
)
from exp.v28_unified.student_models import PixelBridgeV28, build_student_model
from exp.v28_unified.train_student import EVAL_LEVELS_V28, train_student_arm
from src.eval.metrics import paired_cluster_ci
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import read_json, sanitize_cuda_env, save_json, seed_all, utc


_PUBLIC_METHODS = frozenset({
    "AdaIN", "AesPA", "CUT", "Hist2Style", "SaMAM", "SaMST",
    "StyleGallery", "StyleID", "StyleShot", "StyTr2", "WCT",
})


def _literature_points() -> list[dict]:
    ledger = ROOT / "docs/performance/metrics.csv"
    if not ledger.exists():
        return []
    with ledger.open(encoding="utf-8-sig", newline="") as f:
        return [
            {"name": r["arm"], "dino_c": float(r["dino_c"]), "dino_s": float(r["dino_s"]),
             "csd_top1": float(r["target_top1"])}
            for r in csv.DictReader(f)
            if r["arm"] in _PUBLIC_METHODS and r["protocol"] == "q2a_oracle_cross600"
            and r["n_cross"] == "600" and r["is_diagnostic"].lower() == "false"
            and all(r[k] for k in ("dino_c", "dino_s", "target_top1"))
        ]


def _frontier_indices(points: list[tuple[float, float]]) -> list[int]:
    ordered = sorted(range(len(points)), key=lambda i: points[i], reverse=True)
    front, best_y = [], float("-inf")
    for i in ordered:
        if points[i][1] > best_y:
            front.append(i)
            best_y = points[i][1]
    return sorted(front, key=lambda i: points[i][0])


def run_phase3(device: str = "cuda") -> dict:
    """Phase 3: 4-arm student transfer & dead-zone repair on Pool A (4k steps each)."""
    pool_a = OUT / "pools/pool_A.pt"
    assert pool_a.exists(), f"Missing Pool A: {pool_a}"

    arms_cfg = [
        ("M0_flat_2tier_base", dict(arch="flat", interp_mode="2tier", loss_mode="base")),
        ("M1_flat_3tier_anch", dict(arch="flat", interp_mode="3tier", loss_mode="base")),
        ("M2_unet_2tier_cal", dict(arch="unet", interp_mode="2tier", loss_mode="calibrated")),
        ("M3_unet_3tier_cal", dict(arch="unet", interp_mode="3tier", loss_mode="calibrated")),
    ]
    results = {}
    for name, kw in arms_cfg:
        results[name] = train_student_arm(name, pool_a, updates=2800, eval_every=700, device=device, **kw)

    # Pick Stu* (prefer M3 if monotonic & strong, else best balanced/strong score)
    def _score(r: dict) -> float:
        a = r["arms"]
        return (
            0.40 * a["l100"]["csd_top1"] + 0.30 * a["l075"]["csd_top1"]
            + 0.30 * a["l040"]["csd_top1"] + 0.30 * a["l100"]["dino_s"]
            + (0.05 if r["acceptance"]["monotonic_dino_s"] else -0.05)
        )

    best_name = max(results, key=lambda k: _score(results[k]))
    summary = {
        "updated_utc": utc(),
        "winner": best_name,
        "arms": {k: {
            "l040": v["arms"]["l040"],
            "l075": v["arms"]["l075"],
            "l100": v["arms"]["l100"],
            "wrong": v["arms"]["wrong"],
            "acceptance": v["acceptance"],
        } for k, v in results.items()},
    }
    save_json(OUT / "phase3_summary.json", summary)
    print(f"[Phase 3] Winner Stu* = {best_name}", flush=True)
    return summary


def run_phase4(stu_star_arch: str = "unet", stu_star_interp: str = "3tier", stu_star_loss: str = "calibrated", device: str = "cuda") -> dict:
    """Phase 4: 6-arm teacher duel with frozen Stu* (4k steps each, reusing M3)."""
    pool_old = UNIFIED_OLD_OUT / "pairs.pt"
    pool_a = OUT / "pools/pool_A.pt"
    pool_b = OUT / "pools/pool_B.pt"
    pool_c = OUT / "pools/pool_C.pt"

    results = {}
    # 1. T_old (old 6k ICME_Q32 cache)
    results["T_old"] = train_student_arm(
        "P4_T_old", pool_old, arch=stu_star_arch, interp_mode="2tier", loss_mode=stu_star_loss,
        updates=2800, eval_every=700, device=device,
    )
    # 2. T_rand (T1 multi-expert random selection without referee-free gate/ranking)
    results["T_rand"] = train_student_arm(
        "P4_T_rand", pool_a, arch=stu_star_arch, interp_mode=stu_star_interp, loss_mode=stu_star_loss,
        pool_variant="rand", updates=2800, eval_every=700, device=device,
    )
    # 3. Reuse M3 (Pool A)
    m3_res_p = OUT / "students/M3_unet_3tier_cal/results.json"
    if m3_res_p.exists() and stu_star_arch == "unet" and stu_star_interp == "3tier":
        results["Pool_A_M3"] = read_json(m3_res_p)
    else:
        results["Pool_A_M3"] = train_student_arm(
            "P4_Pool_A", pool_a, arch=stu_star_arch, interp_mode=stu_star_interp, loss_mode=stu_star_loss,
            updates=2800, eval_every=700, device=device,
        )
    # 4. T_opt (Pool B zero-legacy optimizer)
    results["T_opt"] = train_student_arm(
        "P4_T_opt", pool_b, arch=stu_star_arch, interp_mode=stu_star_interp, loss_mode=stu_star_loss,
        updates=2800, eval_every=700, device=device,
    )

    # Pick stronger main pool between Pool A (M3) and Pool B (T_opt) for TeleStyle curriculum
    def _rank(r: dict) -> float:
        a = r["arms"]
        er_u = (a["l100"]["hits"]["ER"][0] + a["l100"]["hits"]["Uki"][0]) / 240.0
        return 0.45 * a["l100"]["csd_top1"] + 0.25 * a["l075"]["csd_top1"] + 0.15 * a["l040"]["csd_top1"] + 0.15 * er_u

    main_pool_key = "Pool_A_M3" if _rank(results["Pool_A_M3"]) >= _rank(results["T_opt"]) else "T_opt"
    main_pool_path = pool_a if main_pool_key == "Pool_A_M3" else pool_b
    print(f"[Phase 4] Main base pool for T_rev curriculum: {main_pool_key} ({main_pool_path.name})", flush=True)

    # 5. T_rev (Main pool + 20% Pool C real ER/Uki in last 25% steps at 0.3x LR)
    results["T_rev"] = train_student_arm(
        "P4_T_rev", main_pool_path, arch=stu_star_arch, interp_mode=stu_star_interp, loss_mode=stu_star_loss,
        pool_c_path=pool_c, pool_c_field="z_real", updates=2800, eval_every=700, device=device,
    )
    # 6. T_rev_c1 (Same curriculum, forward teacher endpoints z_fwd)
    results["T_rev_c1"] = train_student_arm(
        "P4_T_rev_c1", main_pool_path, arch=stu_star_arch, interp_mode=stu_star_interp, loss_mode=stu_star_loss,
        pool_c_path=pool_c, pool_c_field="z_fwd", updates=2800, eval_every=700, device=device,
    )
    # 7. T_rev_c2 (Same curriculum, shuffled same-style real endpoints z_shuf)
    results["T_rev_c2"] = train_student_arm(
        "P4_T_rev_c2", main_pool_path, arch=stu_star_arch, interp_mode=stu_star_interp, loss_mode=stu_star_loss,
        pool_c_path=pool_c, pool_c_field="z_shuf", updates=2800, eval_every=700, device=device,
    )

    best_arm = max(("Pool_A_M3", "T_opt", "T_rev"), key=lambda k: _rank(results[k]))
    summary = {
        "updated_utc": utc(),
        "main_base_pool": main_pool_key,
        "winner": best_arm,
        "arms": {k: {
            "l040": v["arms"]["l040"],
            "l075": v["arms"]["l075"],
            "l100": v["arms"]["l100"],
            "wrong": v["arms"]["wrong"],
            "acceptance": v["acceptance"],
        } for k, v in results.items()},
    }
    save_json(OUT / "phase4_summary.json", summary)
    print(f"[Phase 4] Winner Pool* arm = {best_arm}", flush=True)
    return summary


def build_pool_v22_clean(
    pool_dir: str = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"),
    *,
    shard_size: int = 256,
    device: str = "cuda",
) -> Path:
    """Build 7,500-pair coherent two-tier pool from single neural teacher V22 Peak1500 (+ 300-sample audit)."""
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v20.submode_patchswd import build_or_load_core_knn_bank
    from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
    from exp.v27_single.unified.teacher_gate import task_plan
    from exp.v28_unified import SEED_PLAN, atomic_save, recipe_hash
    from exp.v28_unified.data_map import resolve_train_rgb_paths
    from exp.v28_unified.phase1_race import evaluate_150_referee
    from exp.v28_unified.teacher_routes import V22CleanTeacher

    out_p = OUT / "pools/pool_V22_clean.pt"
    r_hash = recipe_hash({
        "pool": "V22_clean_v1",
        "peak_hiC": V22CleanTeacher.PEAK_HIC,
        "peak_hiS": V22CleanTeacher.PEAK_HIS,
        "seed": SEED_PLAN,
    })
    if out_p.exists():
        ck = torch.load(out_p, map_location="cpu", weights_only=False)
        if ck.get("recipe_hash") == r_hash and len(ck["src"]) == 7500:
            print(f"[Phase 4.5 Pool V22] Reusing {out_p}", flush=True)
            return out_p

    pack = build_or_load_v21_cache(pool_dir, device=device)
    bundle = pack["bundle"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=device)
    tr = bundle["v18_cache"]["train"]
    paths_4400 = resolve_train_rgb_paths(tr["ids"], tr["sid"], require_full=True)

    selected, _, _ = task_plan(tr)
    src = selected.repeat_interleave(5)
    tid = torch.arange(5).repeat(len(selected))
    same = tr["sid"][src] == tid
    cross_ids = (~same).nonzero().flatten()

    teacher = V22CleanTeacher(bundle, bank, device=device)
    shard_dir = OUT / "pools/shards_pool_V22_clean"
    shard_dir.mkdir(parents=True, exist_ok=True)
    y16_all = coarse_y16_lcn(tr["y_64"].float())
    shards = []
    t0 = time.monotonic()

    for sh_i, ids in enumerate(cross_ids.split(shard_size)):
        sp = shard_dir / f"shard_{sh_i:03d}.pt"
        if sp.exists():
            pl = torch.load(sp, map_location="cpu", weights_only=False)
            if pl.get("recipe_hash") == r_hash and torch.equal(pl["ids"], ids):
                shards.append(pl)
                continue

        gen = torch.Generator().manual_seed(SEED_PLAN + 1000 + sh_i)
        eps = torch.randn((len(ids), 4, 64, 64), generator=gen).half()
        s_idx = src[ids]
        t_idx = tid[ids]
        hi_c, hi_s = teacher.generate_two_tiers(
            tr["z"][s_idx],
            y16_all[s_idx],
            tr["dino_16"][s_idx],
            tr["c_64"][s_idx],
            bundle["fnd"]["tr_fib"][s_idx],
            t_idx,
            eps,
            src_idx=s_idx,
            batch=32,
        )
        pl = {"recipe_hash": r_hash, "ids": ids, "hiC": hi_c, "hiS": hi_s}
        atomic_save(pl, sp)
        shards.append(pl)
        print(f"[Pool V22] shard {sh_i+1}: {min((sh_i+1)*shard_size, len(cross_ids))}/{len(cross_ids)} "
              f"(elapsed={time.monotonic()-t0:.0f}s)", flush=True)

    hi_c_full = tr["z"][src].half().clone()
    hi_s_full = tr["z"][src].half().clone()
    for pl in shards:
        hi_c_full[pl["ids"]] = pl["hiC"]
        hi_s_full[pl["ids"]] = pl["hiS"]

    pool_v22 = {"recipe_hash": r_hash, "src": src, "tid": tid, "same": same, "hiC": hi_c_full, "hiS": hi_s_full}
    atomic_save(pool_v22, out_p)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # Run disjoint 300-sample audit matching Phase 2 seed for apples-to-apples comparison
    dec_fp16 = load_s10_decoder(device)
    gen = torch.Generator().manual_seed(SEED_PLAN + 999)
    cross_mask = ~same
    picks = []
    for s in range(5):
        cand = (cross_mask & (tid == s)).nonzero().flatten()
        picks.append(cand[torch.randperm(len(cand), generator=gen)[:60]])
    idx_300 = torch.cat(picks, dim=0)
    audit = evaluate_150_referee(
        dec_fp16, paths_4400, src[idx_300], tid[idx_300],
        {"V22_clean_hiC": hi_c_full[idx_300], "V22_clean_hiS": hi_s_full[idx_300]},
        device=device,
    )
    del dec_fp16
    gc.collect()
    torch.cuda.empty_cache()
    save_json(OUT / "phase4_5_pool_audit_300.json", {"updated_utc": utc(), "elapsed_s": round(time.monotonic() - t0, 1), "audit": audit})
    print(f"[Pool V22] COMPLETE & AUDITED -> {out_p}", flush=True)
    return out_p


def run_phase4_5(
    pool_dir: str = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"),
    device: str = "cuda",
) -> dict:
    """Phase 4.5: Single consistent strong teacher (V22 Peak1500) + continuous 2tier UNet + hiSWD."""
    pool_v22 = build_pool_v22_clean(pool_dir, device=device)
    pool_old = UNIFIED_OLD_OUT / "pairs.pt"
    pool_c = OUT / "pools/pool_C.pt"

    base_kw = dict(arch="unet", interp_mode="2tier", updates=2800, eval_every=700, curriculum_steps=800, device=device)
    results = {}
    results["P4_5_V22_clean"] = train_student_arm("P4_5_V22_clean", pool_v22, loss_mode="calibrated", **base_kw)
    results["P4_5_V22_rev"] = train_student_arm("P4_5_V22_rev", pool_v22, pool_c_path=pool_c, loss_mode="calibrated", **base_kw)
    results["P4_5_Old_rev"] = train_student_arm("P4_5_Old_rev", pool_old, pool_c_path=pool_c, loss_mode="calibrated", **base_kw)

    def _score(r: dict) -> float:
        a = r["arms"]
        er_u = (a["l100"]["hits"]["ER"][0] + a["l100"]["hits"]["Uki"][0]) / 240.0
        hole_ok = 1.0 if r["acceptance"].get("monotonic_no_l060_hole", True) else -0.25
        wrong_ok = 0.0 if a["wrong"]["csd_top1"] <= 0.025 else -0.50
        return (
            0.35 * a["l100"]["csd_top1"] + 0.25 * a["l075"]["csd_top1"] + 0.15 * a["l040"]["csd_top1"]
            + 0.15 * er_u + 0.20 * a["l100"]["dino_s"] + hole_ok + wrong_ok
        )

    v22_base = max(("P4_5_V22_clean", "P4_5_V22_rev"), key=lambda k: _score(results[k]))
    v22_c_path = pool_c if v22_base == "P4_5_V22_rev" else None
    print(f"[Phase 4.5] Building P4_5_V22_hiSWD on winner base: {v22_base}", flush=True)
    results["P4_5_V22_hiSWD"] = train_student_arm(
        "P4_5_V22_hiSWD", pool_v22, pool_c_path=v22_c_path, loss_mode="hiswd", **base_kw
    )

    winner = max(results, key=lambda k: _score(results[k]))
    winner_cfg_map = {
        "P4_5_V22_clean": dict(base_pool=str(pool_v22), pool_c=None, loss_mode="calibrated"),
        "P4_5_V22_rev": dict(base_pool=str(pool_v22), pool_c=str(pool_c), loss_mode="calibrated"),
        "P4_5_Old_rev": dict(base_pool=str(pool_old), pool_c=str(pool_c), loss_mode="calibrated"),
        "P4_5_V22_hiSWD": dict(base_pool=str(pool_v22), pool_c=str(v22_c_path) if v22_c_path else None, loss_mode="hiswd"),
    }
    summary = {
        "updated_utc": utc(),
        "hiswd_base": v22_base,
        "winner": winner,
        "winner_config": dict(arch="unet", interp_mode="2tier", curriculum_steps=800, **winner_cfg_map[winner]),
        "arms": {k: {
            "l040": v["arms"]["l040"],
            "l060": v["arms"]["l060"],
            "l075": v["arms"]["l075"],
            "l100": v["arms"]["l100"],
            "l115": v["arms"]["l115"],
            "wrong": v["arms"]["wrong"],
            "acceptance": v["acceptance"],
        } for k, v in results.items()},
    }
    save_json(OUT / "phase4_5_summary.json", summary)
    print(f"[Phase 4.5] Winner for Phase 5 = {winner} ({summary['winner_config']})", flush=True)
    return summary


def unified_knob_schedule(strength: float) -> dict:
    """Two-stage continuous single-knob schedule:

    - s in [0.0, 1.0]: pure 1-NFE forward pass (base_lambda = s, 0 proximal steps).
    - s in (1.0, 2.0]: clamp base_lambda at 1.0 (forbid >1.0 residual extrapolation)
      and drive 1-3 S10-BF16 orthogonal SWD proximal steps with continuous step size,
      anisotropic trust radii, and decaying structural pullback.
    """
    s = float(strength)
    if s <= 1.0:
        return {
            "strength": s,
            "base_lambda": max(0.0, s),
            "u": 0.0,
            "prox_steps": 0,
            "eta": 0.0,
            "w_y": 0.0,
            "w_c": 0.0,
            "rho_y_edge": 0.0,
            "rho_y_low": 0.0,
            "rho_c": 0.0,
            "pullback_w": 0.0,
            "mom_w": 0.0,
            "swd_w": 0.0,
        }
    u = min(1.0, max(0.0, s - 1.0))
    prox_steps = 1 if u <= 0.34 else (2 if u <= 0.67 else 3)
    return {
        "strength": s,
        "base_lambda": 1.0,
        "u": u,
        "prox_steps": prox_steps,
        "eta": 0.055 + 0.045 * u,
        "w_y": 0.22 + 0.30 * u,
        "w_c": 0.95,
        "rho_y_edge": 0.05 + 0.12 * u,
        "rho_y_low": 0.10 + 0.18 * u,
        "rho_c": 0.26 + 0.26 * u,
        "pullback_w": 0.18 * (1.0 - 0.85 * u),
        "mom_w": 0.25 + 0.10 * u,
        "swd_w": 0.15 + 0.25 * u,
    }


def _build_train_vgg_swd_refs(pixel: PixelBridgeV28, tr: dict, device: str) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Precompute per-style orthogonal VGG/Oklab target moments & SWD tokens strictly from train_4400."""
    g_p = torch.Generator(device=device).manual_seed(SEED_TRAIN + 505)
    proj = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=g_p), mode="reduced").Q
                 for c in (64, 128, 256))
    style_moms, style_toks = [], [[] for _ in range(3)]
    for s in range(5):
        s_idx = (tr["sid"] == s).nonzero().flatten()[:24]
        ms = []
        ts_list = [[] for _ in range(3)]
        for sl in s_idx.split(6):
            feat, toks = pixel.encode(tr["z"][sl].to(device).float(), grad=False)
            ms.append(feat)
            for j, tk in enumerate(toks):
                ts_list[j].append(tk)
        style_moms.append(torch.cat(ms, 0).mean(0))
        for j in range(3):
            style_toks[j].append(torch.cat(ts_list[j], 0).mean(0))
    ref_moms = torch.stack(style_moms, 0)
    ref_toks = tuple(torch.stack(tl, 0) for tl in style_toks)
    pixel.scale = ref_moms.std(0, unbiased=False).clamp_min(0.05)
    return ref_moms, ref_toks, proj


@torch.no_grad()
def proximal_swd_polish_eval(
    ckpt_path: Path,
    arch: str,
    n_steps: int,
    tag: str,
    device: str = "cuda",
    *,
    knob_strength: float | None = None,
    wrong: bool = False,
) -> dict:
    """Zero-training S10-BF16 orthogonal SWD proximal refinement at base lambda=1.0 (discrete or continuous s in (1, 2])."""
    m_dir = OUT / "measured" / tag
    if (m_dir / "summary_v28.json").exists():
        row = run_measure_v28(tag, device=device)
        if knob_strength is not None:
            row.update(strength=float(knob_strength), wrong=wrong)
        return row

    from src.utils.protocol import pick_wrong_cross_tid

    sched = unified_knob_schedule(knob_strength) if knob_strength is not None else {
        "strength": 1.0, "base_lambda": 1.0, "prox_steps": n_steps, "eta": 0.065,
        "w_y": 0.20, "w_c": 0.95, "rho_y_edge": 0.05, "rho_y_low": 0.10, "rho_c": 0.28,
        "pullback_w": 0.0, "mom_w": 0.25, "swd_w": 0.15,
    }
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_stats = saved["raw_stats"].to(device)
    priors = {k: v.to(device) for k, v in saved["priors"].items()}
    model = build_student_model(arch, raw_stats.shape[-1], device=device).eval()
    model.load_state_dict(saved["ema"])

    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    q = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["test750"]["q32"]
    tr = load_cache("runs/v18/cache/ccf_pool_cache.pt")["train"]
    tasks, src, _ = tasks_and_sources()
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    labels_all = pick_wrong_cross_tid(sid_all, tid_all) if wrong else tid_all
    eps_fixed = torch.randn(src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    pixel = PixelBridgeV28(device=device)
    ref_moms, ref_toks, proj = _build_train_vgg_swd_refs(pixel, tr, device)

    P_Y, P_C = priors["P_Y"], priors["P_C"]
    dec_fp16 = load_s10_decoder(device)
    wr = RawWriterV28(tag, dec=dec_fp16, device=device)

    for st in range(0, len(tasks), 8):
        sl = slice(st, min(st + 8, len(tasks)))
        zs = src[sl].to(device).float()
        tgt = labels_all[sl].to(device)
        st_t = torch.full((len(zs),), float(sched["base_lambda"]), device=device, dtype=torch.float32)
        if arch == "unet":
            z0 = model(zs, ts["c_64"][sl].to(device).float(), q[sl].to(device).float(),
                       tgt, st_t, raw_stats, priors, noise=eps_fixed[sl].to(device).float())
        else:
            z0 = model(zs, ts["c_64"][sl].to(device).float(), q[sl].to(device).float(),
                       tgt, st_t, raw_stats, priors)

        z = z0.clone()
        for _ in range(int(sched["prox_steps"])):
            with torch.enable_grad():
                zr = z.detach().requires_grad_(True)
                cur_f, cur_t = pixel.encode(zr, grad=True)
                l_mom = F.smooth_l1_loss((cur_f - ref_moms[tgt]) / pixel.scale, torch.zeros_like(cur_f), beta=0.20)
                l_swd = sum(((a @ p).sort(dim=1).values - (b[tgt] @ p).sort(dim=1).values).abs().mean()
                            for a, b, p in zip(cur_t, ref_toks, proj)) / 3.0
                loss_total = sched["mom_w"] * l_mom + sched["swd_w"] * l_swd
                if sched["pullback_w"] > 0.0:
                    loss_total = loss_total + sched["pullback_w"] * (zr - z0).square().mean()
                g = torch.autograd.grad(loss_total, zr)[0].detach()
            g = _smooth_grad_3x3(g, blend=0.25)
            d = aniso_rms_direction(-g, P_Y, P_C, w_y=sched["w_y"], w_c=sched["w_c"])
            z = project_anisotropic_trust(
                z + sched["eta"] * d, z0, P_Y, P_C,
                rho_y_edge=sched["rho_y_edge"], rho_y_low=sched["rho_y_low"], rho_c=sched["rho_c"],
            )
        wr.write_batch(z, tasks[sl])

    wr.finalize(arm=tag, proximal_steps=int(sched["prox_steps"]), strength=float(sched["strength"]), wrong=wrong, same_bypass=False)
    del pixel, model, dec_fp16
    gc.collect()
    torch.cuda.empty_cache()
    row = run_measure_v28(tag, device=device)
    row.update(strength=float(sched["strength"]), wrong=wrong, prox_steps=int(sched["prox_steps"]))
    return row


@torch.no_grad()
def measure_latency_4090(ckpt_path: Path, arch: str, device: str = "cuda") -> dict:
    """Measure b=1 and b=16 model-only and end-to-end (+ S10 decode) latency on 4090."""
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_stats = saved["raw_stats"].to(device)
    priors = {k: v.to(device) for k, v in saved["priors"].items()}
    model = build_student_model(arch, raw_stats.shape[-1], device=device).eval()
    model.load_state_dict(saved["ema"])
    dec = load_s10_decoder(device)

    out = {}
    for bs in (1, 16):
        zs = torch.randn(bs, 4, 64, 64, device=device)
        c64 = torch.randn(bs, 4, 64, 64, device=device)
        q32 = torch.randn(bs, 32, 16, 16, device=device)
        tid = torch.zeros(bs, dtype=torch.long, device=device)
        st = torch.ones(bs, device=device)
        eps = torch.randn_like(zs)

        for _ in range(8):
            _ = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps) if arch == "unet" else model(zs, c64, q32, tid, st, raw_stats, priors)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps) if arch == "unet" else model(zs, c64, q32, tid, st, raw_stats, priors)
        torch.cuda.synchronize()
        model_ms = (time.perf_counter() - t0) * 1000.0 / 30.0

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(20):
            z_o = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps) if arch == "unet" else model(zs, c64, q32, tid, st, raw_stats, priors)
            _ = decode_s10(dec, z_o, batch=bs, output_cpu=False)
        torch.cuda.synchronize()
        e2e_ms = (time.perf_counter() - t1) * 1000.0 / 20.0
        out[f"b{bs}"] = {
            "model_only_ms_total": round(model_ms, 2),
            "model_only_ms_per_img": round(model_ms / bs, 2),
            "e2e_s10_ms_total": round(e2e_ms, 2),
            "e2e_s10_ms_per_img": round(e2e_ms / bs, 2),
            "e2e_img_per_s": round(1000.0 * bs / e2e_ms, 1),
        }
    del model, dec
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def audit_same_style_ab(ckpt_path: Path, arch: str, device: str = "cuda") -> dict:
    """Verify positive non-collapsed brushstroke response to same-style grayscale donors (no FiberEncoder)."""
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_stats = saved["raw_stats"].to(device)
    priors = {k: v.to(device) for k, v in saved["priors"].items()}
    model = build_student_model(arch, raw_stats.shape[-1], device=device).eval()
    model.load_state_dict(saved["ema"])
    dec = load_s10_decoder(device)

    tr = load_cache("runs/v18/cache/ccf_pool_cache.pt")["train"]
    ho = load_cache("runs/v18/cache/ccf_pool_cache.pt")["holdout"]
    fnd = load_cache("runs/v19/cache/v19_foundation.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    q = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["test750"]["q32"]
    tasks, src, _ = tasks_and_sources()

    # Compute normalization stats matching conditioning_stats(tr, fnd, pixel)
    seed_s = torch.Generator().manual_seed(20261127 + 100)
    styles_vecs = []
    for s in range(5):
        ids = (tr["sid"] == s).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), generator=seed_s)[:64]]
        rgba = []
        for idx in ids.split(8):
            rgb = decode_s10(dec, tr["z"][idx].to(device).float(), batch=8, output_cpu=False)
            rgba.append(haar_signature(rgb))
        rgb_s = fnd["tr_fib"][ids].to(device).float()
        styles_vecs.append(torch.cat((rgb_s[:, :27], torch.cat(rgba), rgb_s[:, 27:411]), 1))
    flat = torch.stack(styles_vecs).flatten(0, 1)
    stat_mean = flat.mean(0)
    stat_std = flat.std(0, unbiased=False).clamp_min(0.01)

    out = {}
    for tid in (0, 4):
        pool = (ho["sid"] == tid).nonzero().flatten()[:32]
        sigs, rgbs = [], []
        for sl in pool.split(8):
            rgb = decode_s10(dec, ho["z"][sl].to(device).float(), batch=8, output_cpu=False)
            rgbs.append(rgb)
            sigs.append(band_signature(normalized_gray(rgb)))
        sigs = torch.cat(sigs)
        rgbs = torch.cat(rgbs)
        a, b = divmod(int(torch.cdist(sigs, sigs).argmax()), len(pool))
        ref_delta = sigs[b] - sigs[a]
        ab_idx = pool[[a, b]]

        # Build donor conditioning stats (keeping color/dino constant, varying grayscale Haar signature!)
        base_stat = raw_stats[tid].clone()
        h_a, h_b = haar_signature(rgbs[a:a + 1])[0], haar_signature(rgbs[b:b + 1])[0]
        donor_stats = []
        for h_vec in (h_a, h_b):
            st_vec = raw_stats.clone()
            # Indices [27 : 27 + len(h_vec)] correspond to haar_signature
            st_vec[tid, 27:27 + len(h_vec)] = (h_vec - stat_mean[27:27 + len(h_vec)]) / stat_std[27:27 + len(h_vec)]
            donor_stats.append(st_vec)

        test_ids = [i for i, row in enumerate(tasks) if row["target_style_id"] == tid and row["source_style_id"] != tid][:16]
        t_idx = torch.tensor(test_ids)
        outs = []
        for d_st in donor_stats:
            zs = src[t_idx].to(device).float()
            ti = torch.full((len(t_idx),), tid, device=device)
            lv = torch.ones(len(t_idx), device=device)
            z_g = model(zs, ts["c_64"][t_idx].to(device).float(), q[t_idx].to(device).float(),
                        ti, lv, d_st, priors, noise=torch.zeros_like(zs)) if arch == "unet" else model(
                zs, ts["c_64"][t_idx].to(device).float(), q[t_idx].to(device).float(), ti, lv, d_st, priors)
            outs.append(decode_s10(dec, z_g, batch=8, output_cpu=False))
        xa, xb = outs
        d_tex = band_signature(normalized_gray(xb)) - band_signature(normalized_gray(xa))
        out[STYLES[tid]] = {
            "donor_ids": [ho["ids"][i] for i in ab_idx.tolist()],
            "output_rgb_mae": round(float((xa - xb).abs().mean()), 5),
            "donor_gray_band_l2": round(float(ref_delta.norm()), 4),
            "output_gray_band_l2": round(float(d_tex.norm(dim=-1).mean()), 4),
            "output_to_donor_band_direction_cosine": round(float(F.cosine_similarity(d_tex, ref_delta[None]).mean()), 4),
        }
    del model, dec
    torch.cuda.empty_cache()
    return out


def plot_final_pareto(final_report: dict, out_dir: Path) -> None:
    """Plot 7-point lambda frontier overlaying historical V19-V25 frontier & literature Pareto."""
    literature = _literature_points()
    archived = [
        ("PRISM-1", 0.7440, 0.4431, 0.4433),
        ("R12 g=.55", 0.7723, 0.4564, 0.4250),
        ("V23 HiC", 0.6299, 0.5330, 0.5883),
        ("V20 D2", 0.6050, 0.5672, 0.6050),
        ("V22 Peak1500", 0.5730, 0.5820, 0.7683),
    ]
    pts = [v for k, v in final_report["arms"].items() if k != "wrong" and v["strength"] > 0]
    pts.sort(key=lambda r: r["strength"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    for ax, (metric, ylabel, scale) in zip(
        axes, (("dino_s", "DINO-S ↑", 1.0), ("csd_top1", "CSD Top-1 (%) ↑", 100.0))
    ):
        pub = [(r["dino_c"], scale * r[metric]) for r in literature]
        front = _frontier_indices(pub)
        if front:
            ax.scatter(*zip(*pub), marker="o", s=16, facecolors="none", edgecolors="#666974", zorder=2)
            ax.plot(*zip(*(pub[i] for i in front)), color="#666974", lw=1.2, ls="--", marker="o", ms=3,
                    zorder=2, label="External literature Pareto")
        for i, (name, dc, ds, csd) in enumerate(archived):
            y = ds if metric == "dino_s" else 100.0 * csd
            ax.scatter(dc, y, marker="x", c="#c9722c", s=34, zorder=3, label="Historical V19-V25" if i == 0 else None)
            ax.annotate(name, (dc, y), xytext=(4, 4), textcoords="offset points", color="#c9722c", fontsize=7)
        ax.plot([p["dino_c"] for p in pts], [scale * p[metric] for p in pts],
                marker="D", color="#1b6b52", lw=1.8, ms=5.5, zorder=4, label="PRISM-1-Unified v2 (1 NFE)")
        for p in pts:
            ax.annotate(f"{p['strength']:g}", (p["dino_c"], scale * p[metric]),
                        xytext=(5, 5), textcoords="offset points", color="#1b6b52", fontsize=8)
        ax.grid(alpha=0.18, zorder=0)
        ax.set(xlabel="DINO-C ↑", ylabel=ylabel)
        ax.margins(x=0.10, y=0.15)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("PRISM-1-Unified v2 (Single Ckpt, 1-NFE, No Bypass) · Cross600", fontsize=12, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"pareto_v28.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def export_visual_strip(tag: str, out_png: Path) -> None:
    """Export hard ER, Uki, and Imp->Min visual progression across lambda = 0, 0.40, 0.75, 1.00."""
    tasks, _, _ = tasks_and_sources()
    picks = []
    for s_tgt in (0, 4, 2):
        match = next(t for t in tasks if t["source_style_id"] != t["target_style_id"] and t["target_style_id"] == s_tgt)
        picks.append(match)
    arms = [f"{tag}_l000", f"{tag}_l040", f"{tag}_l075", f"{tag}_l100"]
    cell = 224
    W = len(arms) * (cell + 6) + 10
    H = len(picks) * (cell + 26) + 28
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    for ci, arm in enumerate(arms):
        dr.text((8 + ci * (cell + 6), 6), arm[-4:], fill=(200, 245, 220))
    for ri, t in enumerate(picks):
        y0 = 26 + ri * (cell + 26)
        dr.text((8, y0 + cell + 2), f"{t['source_style'][:4]}->{t['target_style'][:4]}: {t['source_stem'][:22]}", fill=(255, 220, 120))
        for ci, arm in enumerate(arms):
            raw_dir = OUT / "raw" / arm
            for bf in sorted(raw_dir.glob("batch_*.pt")):
                pl = torch.load(bf, map_location="cpu", weights_only=False)
                if t["task_id"] in pl["ids"]:
                    j = list(pl["ids"]).index(t["task_id"])
                    im = Image.fromarray(pl["rgb_u8"][j].permute(1, 2, 0).numpy()).resize((cell, cell))
                    canvas.paste(im, (6 + ci * (cell + 6), y0))
                    break
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def run_phase5(pool_winner: str | None = None, arch: str = "unet", device: str = "cuda") -> dict:
    """Phase 5: 3.5k updates dual-seed finalization from Phase 4.5 winner (strict continuous 2tier UNet)."""
    p45_p = OUT / "phase4_5_summary.json"
    if p45_p.exists():
        p45 = read_json(p45_p)
        cfg = p45["winner_config"]
        pool_winner = p45["winner"]
        base_pool = Path(cfg["base_pool"])
        c_path = Path(cfg["pool_c"]) if cfg.get("pool_c") else None
        kw = dict(
            arch=cfg.get("arch", arch),
            interp_mode=cfg.get("interp_mode", "2tier"),
            loss_mode=cfg.get("loss_mode", "calibrated"),
            curriculum_steps=int(cfg.get("curriculum_steps", 800)),
            updates=3500,
            eval_every=700,
            device=device,
        )
    else:
        raise FileNotFoundError(f"Refusing to run Phase 5 without Phase 4.5 winner summary: {p45_p}")

    rep_s0 = train_student_arm("Final_5k_seed0", base_pool, pool_c_path=c_path, seed=SEED_TRAIN, **kw)
    rep_s1 = train_student_arm("Final_5k_seed1", base_pool, pool_c_path=c_path, seed=SEED_TRAIN + 777, **kw)

    ckpt_s0 = OUT / "students/Final_5k_seed0/Final_5k_seed0_final.pt"
    prox_1 = proximal_swd_polish_eval(ckpt_s0, arch, n_steps=1, tag="Final_5k_prox1", device=device)
    prox_2 = proximal_swd_polish_eval(ckpt_s0, arch, n_steps=2, tag="Final_5k_prox2", device=device)

    latency = measure_latency_4090(ckpt_s0, arch, device=device)
    ab_resp = audit_same_style_ab(ckpt_s0, arch, device=device)

    # Bootstrap CI: seed0 vs seed1, and prox vs 1-NFE
    r_s0_100 = read_json(OUT / "measured/Final_5k_seed0_l100/eval.json")["rows"]
    r_s1_100 = read_json(OUT / "measured/Final_5k_seed1_l100/eval.json")["rows"]
    r_prox1 = read_json(OUT / "measured/Final_5k_prox1/eval.json")["rows"]
    r_prox2 = read_json(OUT / "measured/Final_5k_prox2/eval.json")["rows"]

    plot_final_pareto(rep_s0, OUT)
    export_visual_strip("Final_5k_seed0", OUT / "visual_audit_strip.png")

    audit = {
        "updated_utc": utc(),
        "pool_winner": pool_winner,
        "seed0": rep_s0,
        "seed1": rep_s1,
        "proximal_swd_1step": prox_1,
        "proximal_swd_2step": prox_2,
        "bootstrap_ci": {
            "seed0_minus_seed1_l100": paired_cluster_ci(r_s0_100, r_s1_100),
            "prox1_minus_1nfe_l100": paired_cluster_ci(r_prox1, r_s0_100),
            "prox2_minus_1nfe_l100": paired_cluster_ci(r_prox2, r_s0_100),
        },
        "latency_4090": latency,
        "same_style_ab_response": ab_resp,
    }
    save_json(OUT / "phase5_final_audit.json", audit)
    print(f"[Phase 5] COMPLETE -> {OUT / 'phase5_final_audit.json'}", flush=True)
    return audit


def measure_knob2_latency_4090(ckpt_path: Path, arch: str, device: str = "cuda") -> dict:
    """Measure real 4090 per-image latency across s=1.00 (1-NFE) and s=(1,2] (1-3 proximal steps)."""
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_stats = saved["raw_stats"].to(device)
    priors = {k: v.to(device) for k, v in saved["priors"].items()}
    model = build_student_model(arch, raw_stats.shape[-1], device=device).eval()
    model.load_state_dict(saved["ema"])
    dec = load_s10_decoder(device)
    pixel = PixelBridgeV28(device=device)
    tr = load_cache("runs/v18/cache/ccf_pool_cache.pt")["train"]
    ref_moms, ref_toks, proj = _build_train_vgg_swd_refs(pixel, tr, device)
    P_Y, P_C = priors["P_Y"], priors["P_C"]

    out = {}
    for s_val in (1.25, 1.50, 2.00):
        sched = unified_knob_schedule(s_val)
        n_st = int(sched["prox_steps"])
        key = f"s{int(round(s_val * 100)):03d}_{n_st}step"
        per_bs = {}
        for bs in (1, 8):
            zs = torch.randn(bs, 4, 64, 64, device=device)
            c64 = torch.randn(bs, 4, 64, 64, device=device)
            q32 = torch.randn(bs, 32, 16, 16, device=device)
            tgt = torch.zeros(bs, dtype=torch.long, device=device)
            st_t = torch.ones(bs, device=device)
            eps = torch.randn_like(zs)

            def _once():
                with torch.no_grad():
                    z0 = model(zs, c64, q32, tgt, st_t, raw_stats, priors, noise=eps) if arch == "unet" else model(zs, c64, q32, tgt, st_t, raw_stats, priors)
                z = z0.clone()
                for _ in range(n_st):
                    with torch.enable_grad():
                        zr = z.detach().requires_grad_(True)
                        cf, ct = pixel.encode(zr, grad=True)
                        lm = F.smooth_l1_loss((cf - ref_moms[tgt]) / pixel.scale, torch.zeros_like(cf), beta=0.20)
                        ls = sum(((a @ p).sort(dim=1).values - (b[tgt] @ p).sort(dim=1).values).abs().mean()
                                 for a, b, p in zip(ct, ref_toks, proj)) / 3.0
                        lt = sched["mom_w"] * lm + sched["swd_w"] * ls + sched["pullback_w"] * (zr - z0).square().mean()
                        g = torch.autograd.grad(lt, zr)[0].detach()
                    with torch.no_grad():
                        d = aniso_rms_direction(-_smooth_grad_3x3(g, 0.25), P_Y, P_C, w_y=sched["w_y"], w_c=sched["w_c"])
                        z = project_anisotropic_trust(
                            z + sched["eta"] * d, z0, P_Y, P_C,
                            rho_y_edge=sched["rho_y_edge"], rho_y_low=sched["rho_y_low"], rho_c=sched["rho_c"],
                        )
                with torch.no_grad():
                    _ = decode_s10(dec, z, batch=bs, output_cpu=False)

            for _ in range(3):
                _once()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                _once()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / 10.0
            per_bs[f"b{bs}"] = {
                "e2e_s10_ms_per_img": round(ms / bs, 2),
                "e2e_img_per_s": round(1000.0 * bs / ms, 2),
            }
        out[key] = per_bs

    del pixel, model, dec
    torch.cuda.empty_cache()
    return out


def plot_unified_knob2_pareto(seed0_rep: dict, knob2_arms: dict, out_dir: Path) -> None:
    """Plot full s in [0, 2] unified knob curve: [0, 1] 1-NFE + (1, 2] continuous proximal SWD."""
    literature = _literature_points()
    archived = [
        ("PRISM-1", 0.7440, 0.4431, 0.4433),
        ("R12 g=.55", 0.7723, 0.4564, 0.4250),
        ("V23 HiC", 0.6299, 0.5330, 0.5883),
        ("V20 D2", 0.6050, 0.5672, 0.6050),
        ("V22 Peak1500", 0.5730, 0.5820, 0.7683),
    ]
    base_pts = [seed0_rep["arms"][k] for k in ("l025", "l040", "l060", "l075", "l100")]
    old_extrap = seed0_rep["arms"]["l115"]
    prox_pts = [seed0_rep["arms"]["l100"]] + [
        v for k, v in sorted(knob2_arms.items(), key=lambda kv: kv[1]["strength"]) if k != "wrong"
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), layout="constrained")
    for ax, (metric, ylabel, scale) in zip(
        axes, (("dino_s", "DINO-S ↑", 1.0), ("csd_top1", "CSD Top-1 (%) ↑", 100.0))
    ):
        pub = [(r["dino_c"], scale * r[metric]) for r in literature]
        front = _frontier_indices(pub)
        if front:
            ax.scatter(*zip(*pub), marker="o", s=16, facecolors="none", edgecolors="#666974", zorder=2)
            ax.plot(*zip(*(pub[i] for i in front)), color="#666974", lw=1.2, ls="--", marker="o", ms=3,
                    zorder=2, label="External literature Pareto")
        for i, (name, dc, ds, csd) in enumerate(archived):
            y = ds if metric == "dino_s" else 100.0 * csd
            ax.scatter(dc, y, marker="x", c="#c9722c", s=34, zorder=3, label="Historical V19-V25" if i == 0 else None)
            ax.annotate(name, (dc, y), xytext=(4, 4), textcoords="offset points", color="#c9722c", fontsize=7)

        ax.scatter(old_extrap["dino_c"], scale * old_extrap[metric], marker="x", c="#9ca3af", s=34, zorder=3,
                   label="Old extrap (λ=1.15, replaced)")
        ax.plot([p["dino_c"] for p in base_pts], [scale * p[metric] for p in base_pts],
                marker="D", color="#1b6b52", lw=2.0, ms=5.5, zorder=4, label="Unified s∈[0,1] (1-NFE)")
        for p in base_pts:
            ax.annotate(f"{p['strength']:g}", (p["dino_c"], scale * p[metric]),
                        xytext=(5, 4), textcoords="offset points", color="#1b6b52", fontsize=8)

        ax.plot([p["dino_c"] for p in prox_pts], [scale * p[metric] for p in prox_pts],
                marker="s", color="#b91c48", lw=2.0, ms=5.5, zorder=5, label="Unified s∈(1,2] (1-3 Prox SWD)")
        for p in prox_pts[1:]:
            ax.annotate(f"{p['strength']:g}", (p["dino_c"], scale * p[metric]),
                        xytext=(5, 4), textcoords="offset points", color="#b91c48", fontsize=8)

        ax.grid(alpha=0.18, zorder=0)
        ax.set(xlabel="DINO-C ↑", ylabel=ylabel)
        ax.margins(x=0.10, y=0.15)
    axes[0].legend(loc="best", frameon=False, fontsize=7.5)
    fig.suptitle("PRISM-1-Unified v2 Single-Knob Full Spectrum (s∈[0,1] 1-NFE + s∈(1,2] Proximal SWD) · Cross600",
                 fontsize=11.5, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"pareto_v28_knob2.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def export_knob2_visual_strip(seed0_tag: str, prox_tags: list[str], out_png: Path) -> None:
    """Export s = 0, 0.40, 1.00, 1.50, 2.00 progression strip across hard tasks."""
    tasks, _, _ = tasks_and_sources()
    picks = [next(t for t in tasks if t["source_style_id"] != t["target_style_id"] and t["target_style_id"] == s)
             for s in (0, 4, 2)]
    arms = [(f"{seed0_tag}_l000", "s=0.00"), (f"{seed0_tag}_l040", "s=0.40"), (f"{seed0_tag}_l100", "s=1.00")]
    for pt in prox_tags:
        arms.append((pt, f"s={int(pt[-3:]) / 100.0:.2f}"))
    cell = 200
    W = len(arms) * (cell + 6) + 10
    H = len(picks) * (cell + 26) + 28
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    for ci, (_, lbl) in enumerate(arms):
        dr.text((8 + ci * (cell + 6), 6), lbl, fill=(200, 245, 220))
    for ri, t in enumerate(picks):
        y0 = 26 + ri * (cell + 26)
        dr.text((8, y0 + cell + 2), f"{t['source_style'][:4]}->{t['target_style'][:4]}: {t['source_stem'][:22]}", fill=(255, 220, 120))
        for ci, (arm, _) in enumerate(arms):
            raw_dir = OUT / "raw" / arm
            for bf in sorted(raw_dir.glob("batch_*.pt")):
                pl = torch.load(bf, map_location="cpu", weights_only=False)
                if t["task_id"] in pl["ids"]:
                    j = list(pl["ids"]).index(t["task_id"])
                    im = Image.fromarray(pl["rgb_u8"][j].permute(1, 2, 0).numpy()).resize((cell, cell))
                    canvas.paste(im, (6 + ci * (cell + 6), y0))
                    break
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def run_knob2_frontier(device: str = "cuda") -> dict:
    """Zero-retraining extension: map s in (1, 2] to continuous 1-3 step S10-BF16 SWD polish (no >1.0 extrap)."""
    ckpt_s0 = OUT / "students/Final_5k_seed0/Final_5k_seed0_final.pt"
    assert ckpt_s0.exists(), f"Missing final checkpoint: {ckpt_s0}"
    seed0_rep = read_json(OUT / "students/Final_5k_seed0/results.json")

    knob_arms = {}
    for s_val in (1.25, 1.50, 1.75, 2.00):
        tag = f"Final_5k_s{int(round(s_val * 100)):03d}"
        sched = unified_knob_schedule(s_val)
        knob_arms[f"s{int(round(s_val * 100)):03d}"] = proximal_swd_polish_eval(
            ckpt_s0, "unet", sched["prox_steps"], tag=tag, device=device, knob_strength=s_val, wrong=False,
        )
    knob_arms["wrong"] = proximal_swd_polish_eval(
        ckpt_s0, "unet", unified_knob_schedule(2.0)["prox_steps"], tag="Final_5k_s200_wrong",
        device=device, knob_strength=2.0, wrong=True,
    )

    r_s100 = read_json(OUT / "measured/Final_5k_seed0_l100/eval.json")["rows"]
    r_s150 = read_json(OUT / "measured/Final_5k_s150/eval.json")["rows"]
    r_s200 = read_json(OUT / "measured/Final_5k_s200/eval.json")["rows"]

    latency = measure_knob2_latency_4090(ckpt_s0, "unet", device=device)
    plot_unified_knob2_pareto(seed0_rep, knob_arms, OUT)
    export_knob2_visual_strip("Final_5k_seed0", ["Final_5k_s150", "Final_5k_s200"], OUT / "visual_audit_strip_knob2.png")

    report = {
        "updated_utc": utc(),
        "base_checkpoint": ckpt_s0.name,
        "seed0_l100": seed0_rep["arms"]["l100"],
        "knob2_arms": knob_arms,
        "bootstrap_ci": {
            "s150_minus_s100": paired_cluster_ci(r_s150, r_s100),
            "s200_minus_s100": paired_cluster_ci(r_s200, r_s100),
        },
        "latency_4090_knob2": latency,
        "acceptance": {
            "no_extrap_above_1": all(unified_knob_schedule(s)["base_lambda"] == 1.0 for s in (1.25, 1.50, 1.75, 2.00)),
            "monotonic_dino_s_s1_to_s2": (
                seed0_rep["arms"]["l100"]["dino_s"] <= knob_arms["s125"]["dino_s"]
                <= knob_arms["s150"]["dino_s"] <= knob_arms["s175"]["dino_s"] <= knob_arms["s200"]["dino_s"]
            ),
            "monotonic_top1_s1_to_s2": (
                seed0_rep["arms"]["l100"]["csd_top1"] <= knob_arms["s125"]["csd_top1"]
                <= knob_arms["s150"]["csd_top1"] <= knob_arms["s175"]["csd_top1"] <= knob_arms["s200"]["csd_top1"]
            ),
            "wrong_target_s200": knob_arms["wrong"]["csd_top1"] <= 0.025,
        },
    }
    save_json(OUT / "phase5_knob2_audit.json", report)
    print(f"[Knob2] COMPLETE -> {OUT / 'phase5_knob2_audit.json'}", flush=True)
    return report
