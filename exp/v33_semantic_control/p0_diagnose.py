"""P0 (~1.5h budget, zero retraining):
- P0.1: Sub-school Routing Fix (R0 argmax, R1 train-only CSLS 2*cos-r_k, R2 forced 2nd-nearest).
- P0.2: Decoupling Source Luminance Lock Y16/LCN (Y0 orig, Y1 50% Y16, Y2 0% Y16, Y3 continuous E_src/E_tgt rule).
"""
from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v27_single.bare_q32 import load_cache
from exp.v28_unified.student_models import build_student_model
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v29_biglittle.phase2_tune import default_knob_schedule
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v33_semantic_control import (
    OUT,
    S085_C10_KW,
    SEED_V33,
    STYLES,
    V32_ECORE_CKPT,
    V32_OUT,
    V32_PCORE_CKPT,
    RawWriterV33,
    compute_style_pixel_stats,
    measure_and_summarize,
    paired_ci_vs_baseline,
)
from exp.v33_semantic_control.assets import build_or_load_routing_and_energy, match_cnx_k8_csls
from exp.v33_semantic_control.models import integrate_v33_trajectory
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import read_json, sanitize_cuda_env, save_json, seed_all, utc


def _routing_dist(picks: torch.Tensor) -> dict:
    counts = [int((picks == k).sum().item()) for k in range(8)]
    tot = max(1, len(picks))
    probs = [c / tot for c in counts if c > 0]
    norm_h = (-sum(p * math.log(p) for p in probs)) / math.log(8.0)
    return {
        "counts": counts,
        "active_clusters": sum(1 for c in counts if c > 0),
        "max_cluster_share": round(max(counts) / tot, 4),
        "norm_entropy": round(norm_h, 4),
    }


def _per_cluster_rococo(eval_json: Path, picks_roc: list[int], roc_task_ids: list[str]) -> dict:
    rows = {r["task_id"]: r for r in read_json(eval_json)["rows"]}
    out = {}
    for k in range(8):
        sub = [rows[tid] for tid, cl in zip(roc_task_ids, picks_roc) if cl == k and tid in rows]
        if sub:
            out[f"c{k}"] = {
                "n": len(sub),
                "dino_s": round(sum(r["dino_s"] for r in sub) / len(sub), 4),
                "dino_c": round(sum(r["dino_c"] for r in sub) / len(sub), 4),
                "csd_top1": round(sum(int(r["target_top1"]) for r in sub) / len(sub), 4),
            }
    return out


def _build_p0_context(device: str = "cuda") -> dict:
    assets = build_or_load_cnx_assets(device=device)
    re_assets = build_or_load_routing_and_energy(device=device)
    tasks, z_src, _ = tasks_and_sources()
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")

    ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device).float()
    priors = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device).eval().requires_grad_(False)
    ecore.load_state_dict(ec_sv["ema"], strict=True)

    pc_sv = torch.load(V32_PCORE_CKPT, map_location="cpu", weights_only=False)
    pcore = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=True, normalize_u=True).to(device).eval().requires_grad_(False)
    pcore.load_state_dict(pc_sv["ema"], strict=True)

    return {
        "assets": assets,
        "re": re_assets,
        "tasks": tasks,
        "z_src": z_src,
        "sid": torch.tensor([t["source_style_id"] for t in tasks], dtype=torch.long),
        "tid": torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long),
        "y16": project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float()),
        "c64": ts["c_64"].float(),
        "q32": assets["stiefel"]["test750"]["q32"].float(),
        "cnx_gray": assets["stiefel"]["test750"]["cnx_gray"].float(),
        "cnx_bank": assets["bank"],
        "eps": torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED)),
        "ecore": ecore,
        "raw_stats": raw_stats,
        "priors": priors,
        "pcore": pcore,
        "dec": load_s10_decoder(device),
        "draft_cache": {},
    }


@torch.no_grad()
def _run_p0_arm(
    arm_name: str,
    ctx: dict,
    *,
    sv: float,
    kw: dict,
    route_mode: str = "argmax",
    y16_scale: float | torch.Tensor = 1.0,
    tau0_override: torch.Tensor | None = None,
    device: str = "cuda",
) -> tuple[dict, torch.Tensor]:
    tasks = ctx["tasks"]
    m_dir = OUT / "measured" / arm_name
    cnx_bank = ctx["cnx_bank"]
    csls_r = ctx["re"]["csls_r"]

    # Precompute routing picks for all 750 tasks (for logging & E_tgt consistency)
    all_u, all_picks = [], []
    for st in range(0, len(tasks), 128):
        sl = slice(st, min(st + 128, len(tasks)))
        u_b, pk_b, _ = match_cnx_k8_csls(
            ctx["cnx_gray"][sl].to(device), ctx["tid"][sl].to(device), cnx_bank, csls_r, mode=route_mode,
        )
        all_u.append(u_b.cpu())
        all_picks.append(pk_b.cpu())
    u_750 = torch.cat(all_u, 0)
    picks_750 = torch.cat(all_picks, 0)

    if not (m_dir / "summary_v33.json").exists():
        lam = round(float(kw["draft_lambda"]), 3)
        use_draft = (tau0_override is not None) or (float(kw["tau0"]) > 0.0)
        if use_draft and lam not in ctx["draft_cache"]:
            chunks = []
            for st in range(0, len(tasks), 32):
                sl = slice(st, min(st + 32, len(tasks)))
                chunks.append(ctx["ecore"](
                    ctx["z_src"][sl].to(device).float(), ctx["c64"][sl].to(device).float(),
                    ctx["q32"][sl].to(device).float(), ctx["tid"][sl].to(device),
                    torch.full((sl.stop - sl.start,), lam, device=device),
                    ctx["raw_stats"], ctx["priors"], noise=ctx["eps"][sl].to(device).float(),
                ).cpu().half())
            ctx["draft_cache"][lam] = torch.cat(chunks, 0)
        ze_pool = ctx["draft_cache"].get(lam)

        writer = RawWriterV33(arm_name, dec=ctx["dec"], device=device)
        kw_call = {k: v for k, v in kw.items() if k not in ("draft_lambda", "label")}
        for st in range(0, len(tasks), 32):
            sl = slice(st, min(st + 32, len(tasks)))
            ys = y16_scale[sl].to(device).view(-1, 1, 1, 1) if isinstance(y16_scale, torch.Tensor) else float(y16_scale)
            y16_in = ctx["y16"][sl].to(device).float() * ys
            t0_val = tau0_override[sl].to(device) if tau0_override is not None else float(kw_call["tau0"])
            ze = ze_pool[sl].to(device).float() if (ze_pool is not None and (isinstance(t0_val, torch.Tensor) or t0_val > 0.0)) else None
            kw_b = dict(kw_call)
            kw_b["tau0"] = t0_val
            zo = integrate_v33_trajectory(
                ctx["pcore"], ze, ctx["eps"][sl].to(device).float(),
                y16_in, ctx["q32"][sl].to(device).float(),
                ctx["tid"][sl].to(device), u_750[sl].to(device).float(),
                steps=6, solver="dpm2m", **kw_b,
            )
            writer.write_batch(zo, tasks[sl])
        writer.finalize(arm=arm_name, s=sv, route_mode=route_mode, schedule=kw)

    res = measure_and_summarize(arm_name, device=device)
    print(
        f"[P0] {arm_name:24s} | C={res['dino_c']:.4f} S={res['dino_s']:.4f} Top5={res['dino_s_top5']:.4f} "
        f"CSLS={res['csls_k10']:.4f} Top1={res['csd_top1']*100:5.2f}% LP={res['lpips']:.4f} | "
        f"S_tgt:{res['per_tgt_dino_s']}",
        flush=True,
    )
    return res, picks_750


@torch.no_grad()
def run_p0(device: str = "cuda") -> dict:
    sanitize_cuda_env()
    seed_all(SEED_V33)
    t0 = time.monotonic()

    ctx = _build_p0_context(device=device)
    tasks = ctx["tasks"]
    roc_idx = [i for i, t in enumerate(tasks) if t["target_style_id"] == 3 and t["source_style_id"] != 3]
    roc_task_ids = [tasks[i]["task_id"] for i in roc_idx]

    # --- P0.1: Sub-school Routing Fix ---
    s150_kw = default_knob_schedule(1.50, use_gamma=True)
    p01_arms = {}
    p01_picks = {}
    for arm, rmode in (("P0_1_R0_argmax_s150", "argmax"), ("P0_1_R1_csls_s150", "csls"), ("P0_1_R2_second_s150", "second")):
        res, picks = _run_p0_arm(arm, ctx, sv=1.50, kw=s150_kw, route_mode=rmode, device=device)
        roc_p = picks[roc_idx]
        res["rococo_routing"] = _routing_dist(roc_p)
        res["rococo_per_cluster"] = _per_cluster_rococo(OUT / "measured" / arm / "eval.json", roc_p.tolist(), roc_task_ids)
        p01_arms[arm] = res
        p01_picks[arm] = picks

    # Also test R1 CSLS on s=0.85_c10 (main working point) vs R0
    r0_85, _ = _run_p0_arm("P0_1_R0_argmax_s085", ctx, sv=0.85, kw=S085_C10_KW, route_mode="argmax", device=device)
    r1_85, pk1_85 = _run_p0_arm("P0_1_R1_csls_s085", ctx, sv=0.85, kw=S085_C10_KW, route_mode="csls", device=device)
    r1_85["rococo_routing"] = _routing_dist(pk1_85[roc_idx])

    ci_r1_s150 = paired_ci_vs_baseline(OUT / "measured/P0_1_R1_csls_s150/eval.json", OUT / "measured/P0_1_R0_argmax_s150/eval.json")
    ci_r2_s150 = paired_ci_vs_baseline(OUT / "measured/P0_1_R2_second_s150/eval.json", OUT / "measured/P0_1_R0_argmax_s150/eval.json")
    ci_r1_s085 = paired_ci_vs_baseline(OUT / "measured/P0_1_R1_csls_s085/eval.json", OUT / "measured/P0_1_R0_argmax_s085/eval.json")

    roc_delta_s150 = p01_arms["P0_1_R1_csls_s150"]["per_tgt_dino_s"]["Roc"] - p01_arms["P0_1_R0_argmax_s150"]["per_tgt_dino_s"]["Roc"]
    other_deltas_s150 = [
        p01_arms["P0_1_R1_csls_s150"]["per_tgt_dino_s"][k] - p01_arms["P0_1_R0_argmax_s150"]["per_tgt_dino_s"][k]
        for k in ("Ear", "Imp", "Min", "Uki")
    ]
    dc_delta_s150 = p01_arms["P0_1_R1_csls_s150"]["dino_c"] - p01_arms["P0_1_R0_argmax_s150"]["dino_c"]
    p01_pass = (roc_delta_s150 >= 0.03) and (min(other_deltas_s150) >= -0.015) and (dc_delta_s150 >= -0.005)
    # Also check R1 vs R0 at s085 for adoption at working point
    roc_delta_s085 = r1_85["per_tgt_dino_s"]["Roc"] - r0_85["per_tgt_dino_s"]["Roc"]
    p01_s085_net_positive = (r1_85["dino_s"] >= r0_85["dino_s"]) and (r1_85["dino_c"] - r0_85["dino_c"] >= -0.005)

    # --- P0.2: Decoupling Source Luminance Lock Y16/LCN ---
    # Compute Y3 adaptive scaling continuously from E_src / E_tgt (NO style-name branching)
    _, pk_y3_base, _ = match_cnx_k8_csls(
        ctx["cnx_gray"].to(device), ctx["tid"].to(device), ctx["cnx_bank"], ctx["re"]["csls_r"], mode="argmax",
    )
    e_src = ctx["re"]["E_src_750"].float().cpu()
    e_tgt = ctx["re"]["sub_E_tgt"].float().cpu()[ctx["tid"].cpu(), pk_y3_base.cpu()]
    eps_e = 1.0e-4
    d_log = torch.log((e_src + eps_e) / (e_tgt + eps_e))
    tanh_pos = torch.tanh(d_log.clamp_min(0.0))
    g_y3 = (1.0 - 0.7 * tanh_pos).clamp(0.0, 1.0)
    tau0_y3b = (float(S085_C10_KW["tau0"]) - 0.12 * tanh_pos).clamp(0.0, 1.0)

    p02_arms = {}
    p02_pix = {}
    p02_specs = [
        ("P0_2_Y0_orig_s085", 0.85, S085_C10_KW, 1.0, None),
        ("P0_2_Y1_half_s085", 0.85, S085_C10_KW, 0.5, None),
        ("P0_2_Y2_off_s085", 0.85, S085_C10_KW, 0.0, None),
        ("P0_2_Y3a_adapt_s085", 0.85, S085_C10_KW, g_y3, None),
        ("P0_2_Y3b_adtau_s085", 0.85, S085_C10_KW, g_y3, tau0_y3b),
    ]
    s100_kw = default_knob_schedule(1.00, use_gamma=True)
    tau0_y3b_100 = (float(s100_kw["tau0"]) - 0.12 * tanh_pos).clamp(0.0, 1.0)
    p02_specs += [
        ("P0_2_Y0_orig_s100", 1.00, s100_kw, 1.0, None),
        ("P0_2_Y2_off_s100", 1.00, s100_kw, 0.0, None),
        ("P0_2_Y3b_adtau_s100", 1.00, s100_kw, g_y3, tau0_y3b_100),
    ]

    for arm, sv, kw, ys, t_ov in p02_specs:
        res, _ = _run_p0_arm(arm, ctx, sv=sv, kw=kw, route_mode="argmax", y16_scale=ys, tau0_override=t_ov, device=device)
        pix = compute_style_pixel_stats(OUT / "raw" / arm, target_style_id=2, cross_only=True)
        res["min_pixel_stats"] = pix
        p02_arms[arm] = res
        p02_pix[arm] = pix
        print(f"    [P0.2 Min Pixel] {arm:24s} | sat={pix['sat_hsv']:.4f} chroma={pix['chroma_oklab']:.4f} gstd={pix['gray_std']:.4f} tv={pix['tv_grad']:.4f} | Min_S={res['per_tgt_dino_s']['Min']:.4f} Min_C={res['per_tgt_dino_c']['Min']:.4f}", flush=True)

    # Check P0.2 Y3 vs Y0 at matched DINO-C
    ci_y2_s085 = paired_ci_vs_baseline(OUT / "measured/P0_2_Y2_off_s085/eval.json", OUT / "measured/P0_2_Y0_orig_s085/eval.json")
    ci_y3b_s085 = paired_ci_vs_baseline(OUT / "measured/P0_2_Y3b_adtau_s085/eval.json", OUT / "measured/P0_2_Y0_orig_s085/eval.json")

    del ctx
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "updated_utc": utc(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "p0_1_routing": {
            "s150": p01_arms,
            "s085": {"R0": r0_85, "R1_csls": r1_85},
            "ci_r1_vs_r0_s150": ci_r1_s150,
            "ci_r2_vs_r0_s150": ci_r2_s150,
            "ci_r1_vs_r0_s085": ci_r1_s085,
            "rococo_delta_s150": round(roc_delta_s150, 4),
            "rococo_delta_s085": round(roc_delta_s085, 4),
            "pass_gate_s150": bool(p01_pass),
            "s085_net_positive": bool(p01_s085_net_positive),
        },
        "p0_2_y16_luma": {
            "arms": p02_arms,
            "ci_y2_vs_y0_s085": ci_y2_s085,
            "ci_y3b_vs_y0_s085": ci_y3b_s085,
            "y3_g_mean_per_style": {STYLES[s][:3]: round(float(g_y3[ctx_i].mean()), 4) for s, ctx_i in [(s, [i for i, t in enumerate(tasks) if t["target_style_id"] == s]) for s in range(5)]},
        },
    }
    save_json(OUT / "p0_report.json", report)
    print(f"[P0 DONE] in {report['elapsed_s']:.1f}s | P0.1 gate_s150={p01_pass} (Roc delta={roc_delta_s150:+.4f}) | s085_pos={p01_s085_net_positive}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="CLEAVE V33 P0 Zero-Retraining Diagnostics")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_p0(device=args.device)


if __name__ == "__main__":
    main()
