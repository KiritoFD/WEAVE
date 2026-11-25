"""Zero-retraining P2_S2 CFG-Rebalance Probe + Bottleneck Attribution Diagnostic (~8 min on 4090).

Tests whether re-balancing style-vs-content CFG weights (raising ws/wsp, lowering wc/lscale)
on the trained P2_S2_hi32_pert30 model converts its +0.07 DINO-C surplus into a true Pareto push,
and records per-style breakdowns vs V32 baseline for data/capacity diagnosis.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v27_single.bare_q32 import load_cache
from exp.v28_unified.student_models import build_student_model
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control import (
    OUT,
    SEED_V33,
    V32_ECORE_CKPT,
    V32_OUT,
    V32_PCORE_CKPT,
    RawWriterV33,
    measure_and_summarize,
)
from exp.v33_semantic_control.assets import build_or_load_q64_hi32_cache
from exp.v33_semantic_control.models import PCoreWithSemanticControl, SemanticControlAdapter, integrate_v33_trajectory
from exp.v33_semantic_control.p3_joint_verify import _interp_v32_s_at_c, _v32_frontier_points
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import read_json, sanitize_cuda_env, save_json, seed_all, utc


@torch.no_grad()
def run_p2_cfg_probe(device: str = "cuda") -> dict:
    sanitize_cuda_env()
    seed_all(SEED_V33 + 600)
    t0 = time.monotonic()

    s2_ckpt = OUT / "p2/P2_S2_hi32_pert30/P2_S2_hi32_pert30_s3500.pt"
    if not s2_ckpt.exists():
        raise FileNotFoundError(f"Missing P2_S2 checkpoint: {s2_ckpt}")

    assets = build_or_load_cnx_assets(device=device)
    q64_cache = build_or_load_q64_hi32_cache(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]

    pcore = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=True, normalize_u=True).to(device)
    pcore.load_state_dict(torch.load(V32_PCORE_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    sv = torch.load(s2_ckpt, map_location="cpu", weights_only=False)
    ctrl = SemanticControlAdapter(d_q16=int(sv.get("d_q16", 32)), d_q32=32).to(device)
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
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    y16_750 = project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    c64_750 = ts["c_64"].float()
    q16_750 = stiefel["test750"]["q32"].float()
    q32_hi_750 = q64_cache["test750"]["q32_hi"].float()
    cnx_gray_750 = stiefel["test750"]["cnx_gray"].float()
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    probes = [
        ("P2_S2_cfgA_t000_sb", {"draft_lambda": 1.00, "tau0": 0.00, "ws": 7.20, "wc": 0.40, "wsp": 5.50, "rescale_phi": 0.48, "gamma": 0.25, "lscale": 0.50}),
        ("P2_S2_cfgB_t035_sb", {"draft_lambda": 1.00, "tau0": 0.35, "ws": 5.80, "wc": 0.80, "wsp": 4.20, "rescale_phi": 0.35, "gamma": 0.50, "lscale": 0.62}),
        ("P2_S2_cfgC_t050_sb", {"draft_lambda": 1.00, "tau0": 0.50, "ws": 5.50, "wc": 1.00, "wsp": 3.80, "rescale_phi": 0.30, "gamma": 0.65, "lscale": 0.68}),
        ("P2_S2_cfgD_t065_sb", {"draft_lambda": 0.99, "tau0": 0.65, "ws": 5.00, "wc": 1.40, "wsp": 3.20, "rescale_phi": 0.20, "gamma": 0.80, "lscale": 0.76}),
        ("P2_S2_cfgE_t075_sb", {"draft_lambda": 0.98, "tau0": 0.75, "ws": 4.80, "wc": 1.80, "wsp": 2.60, "rescale_phi": 0.00, "gamma": 0.90, "lscale": 0.84}),
    ]

    v32_pts = _v32_frontier_points()
    draft_cache: dict[float, torch.Tensor] = {}
    out = {}

    for tag, kw in probes:
        m_dir = OUT / "measured" / tag
        if not (m_dir / "summary_v33.json").exists():
            lam = round(float(kw["draft_lambda"]), 3)
            if kw["tau0"] > 0.0 and lam not in draft_cache:
                chunks = []
                for st in range(0, len(tasks), 32):
                    sl = slice(st, min(st + 32, len(tasks)))
                    chunks.append(ecore(
                        z_src[sl].to(device).float(), c64_750[sl].to(device).float(),
                        q16_750[sl].to(device).float(), tid_750[sl].to(device),
                        torch.full((sl.stop - sl.start,), lam, device=device),
                        raw_stats, priors, noise=eps_750[sl].to(device).float(),
                    ).cpu().half())
                draft_cache[lam] = torch.cat(chunks, 0)
            ze_pool = draft_cache.get(lam)

            writer = RawWriterV33(tag, dec=dec, device=device)
            kw_call = {k: v for k, v in kw.items() if k != "draft_lambda"}
            for st in range(0, len(tasks), 32):
                sl = slice(st, min(st + 32, len(tasks)))
                ti = tid_750[sl].to(device)
                u_st = match_cnx_k8_submode(cnx_gray_750[sl].to(device).float(), ti, cnx_bank)
                ze = ze_pool[sl].to(device).float() if kw["tau0"] > 0.0 else None
                zo = integrate_v33_trajectory(
                    model, ze, eps_750[sl].to(device).float(),
                    y16_750[sl].to(device).float(), q16_750[sl].to(device).float(),
                    ti, u_st, steps=6, solver="dpm2m", q32_hi=q32_hi_750[sl].to(device).float(), **kw_call,
                )
                writer.write_batch(zo, tasks[sl])
            writer.finalize(arm=tag, schedule=kw)

        res = measure_and_summarize(tag, device=device)
        s_v32 = _interp_v32_s_at_c(res["dino_c"], v32_pts)
        res["v32_s_at_same_c"] = round(s_v32, 4)
        res["delta_s_vs_v32"] = round(res["dino_s"] - s_v32, 4)
        out[tag] = res
        print(
            f"[CFG Probe] {tag:22s} | C={res['dino_c']:.4f} S={res['dino_s']:.4f} (V32={s_v32:.4f}, dS={res['delta_s_vs_v32']:+.4f}) "
            f"Top5={res['dino_s_top5']:.4f} Top1={res['csd_top1']*100:5.2f}% LP={res['lpips']:.4f}",
            flush=True,
        )

    del model, pcore, ctrl, ecore, dec
    gc.collect()
    torch.cuda.empty_cache()

    rep = {
        "updated_utc": utc(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "probes": out,
        "max_delta_s_vs_v32": max(v["delta_s_vs_v32"] for v in out.values()),
    }
    save_json(OUT / "p2_cfg_probe_report.json", rep)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="CLEAVE V33 P2_S2 CFG Rebalance Probe")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_p2_cfg_probe(device=args.device)


if __name__ == "__main__":
    main()
