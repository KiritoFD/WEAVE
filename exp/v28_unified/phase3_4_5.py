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


@torch.no_grad()
def proximal_swd_polish_eval(
    ckpt_path: Path,
    arch: str,
    n_steps: int,
    tag: str,
    device: str = "cuda",
) -> dict:
    """Zero-training 1-2 step inference-time S10-BF16 orthogonal SWD proximal refinement at lambda=1.0."""
    m_dir = OUT / "measured" / tag
    if (m_dir / "summary_v28.json").exists():
        return run_measure_v28(tag, device=device)

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_stats = saved["raw_stats"].to(device)
    priors = {k: v.to(device) for k, v in saved["priors"].items()}
    model = build_student_model(arch, raw_stats.shape[-1], device=device).eval()
    model.load_state_dict(saved["ema"])

    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    q = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["test750"]["q32"]
    tr = load_cache("runs/v18/cache/ccf_pool_cache.pt")["train"]
    tasks, src, _ = tasks_and_sources()
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    eps_fixed = torch.randn(src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    pixel = PixelBridgeV28(device=device)
    # Build per-style orthogonal VGG target moments & SWD tokens strictly from train_4400
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
    ref_moms = torch.stack(style_moms, 0)  # (5, D)
    ref_toks = tuple(torch.stack(tl, 0) for tl in style_toks)  # each (5, 256, C)
    pixel.scale = ref_moms.std(0, unbiased=False).clamp_min(0.05)

    P_Y, P_C = priors["P_Y"], priors["P_C"]
    dec_fp16 = load_s10_decoder(device)
    wr = RawWriterV28(tag, dec=dec_fp16, device=device)

    for st in range(0, len(tasks), 8):
        sl = slice(st, min(st + 8, len(tasks)))
        zs = src[sl].to(device).float()
        tgt = tid_all[sl].to(device)
        st_t = torch.ones(len(zs), device=device, dtype=torch.float32)
        if arch == "unet":
            z0 = model(zs, ts["c_64"][sl].to(device).float(), q[sl].to(device).float(),
                       tgt, st_t, raw_stats, priors, noise=eps_fixed[sl].to(device).float())
        else:
            z0 = model(zs, ts["c_64"][sl].to(device).float(), q[sl].to(device).float(),
                       tgt, st_t, raw_stats, priors)

        z = z0.clone()
        for _ in range(n_steps):
            with torch.enable_grad():
                zr = z.detach().requires_grad_(True)
                cur_f, cur_t = pixel.encode(zr, grad=True)
                l_mom = F.smooth_l1_loss((cur_f - ref_moms[tgt]) / pixel.scale, torch.zeros_like(cur_f), beta=0.20)
                l_swd = sum(((a @ p).sort(dim=1).values - (b[tgt] @ p).sort(dim=1).values).abs().mean()
                            for a, b, p in zip(cur_t, ref_toks, proj)) / 3.0
                g = torch.autograd.grad(0.25 * l_mom + 0.15 * l_swd, zr)[0].detach()
            g = _smooth_grad_3x3(g, blend=0.25)
            d = aniso_rms_direction(-g, P_Y, P_C, w_y=0.20, w_c=0.95)
            z = project_anisotropic_trust(z + 0.065 * d, z0, P_Y, P_C, rho_y_edge=0.05, rho_y_low=0.10, rho_c=0.28)
        wr.write_batch(z, tasks[sl])

    wr.finalize(arm=tag, proximal_steps=n_steps, strength=1.0, same_bypass=False)
    del pixel, model, dec_fp16
    gc.collect()
    torch.cuda.empty_cache()
    return run_measure_v28(tag, device=device)


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
        c64 = torch.randn(bs, 7, 64, 64, device=device)
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
    """Phase 5: 5k updates dual-seed finalization + proximal polish + complete audit pack."""
    if pool_winner is None:
        p4 = read_json(OUT / "phase4_summary.json")
        pool_winner = p4["winner"]

    pool_a = OUT / "pools/pool_A.pt"
    pool_b = OUT / "pools/pool_B.pt"
    pool_c = OUT / "pools/pool_C.pt"
    main_base = OUT / "pools/pool_A.pt"
    if (OUT / "phase4_summary.json").exists():
        mb = read_json(OUT / "phase4_summary.json").get("main_base_pool", "Pool_A_M3")
        main_base = pool_a if mb == "Pool_A_M3" else pool_b

    kw = dict(arch=arch, interp_mode="3tier", loss_mode="calibrated", updates=3500, eval_every=700, device=device)
    if pool_winner == "T_opt":
        base_pool, c_path = pool_b, None
    elif pool_winner == "T_rev":
        base_pool, c_path = main_base, pool_c
    else:
        base_pool, c_path = pool_a, None

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
