"""F5: E/P Boundary Continuity Audit (H0 Hard Switch vs H1 Latent Blend vs H2 Schedule Interpolation).

Zero training / zero new learnable parameters. Evaluates fixed 150-task stratified subset (30 sources x 5 targets
from tasks_750) cleanly across dense E-Core & P-Core transition regimes, reporting:
  - Adjacent latent RMSE
  - Adjacent RGB LPIPS perceptual continuity
  - DINO-C, DINO-S, CSD Top-1
  - Effective NFE & wall latency overhead
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v27_single.bare_q32 import load_cache
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v35_final import build_pcore_startpoint
from exp.v35_final.common import load_pcore_v35, load_v32_ecore_and_priors, make_v35_schedule
from src.utils import FeatureStack, auto_referee_batch, decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import REPO, read_json, sanitize_cuda_env, save_json, utc


F5_DIR = REPO / "runs/paper_audit/f5_ep_continuity"
P8_REPORT_PATH = REPO / "runs/v35_final/p8_final_report.json"


@torch.no_grad()
def run_f5(device: str = "cuda") -> dict[str, Any]:
    F5_DIR.mkdir(parents=True, exist_ok=True)
    out_p = F5_DIR / "f5_summary.json"
    if out_p.exists():
        return read_json(out_p)

    p8 = read_json(P8_REPORT_PATH)
    fcfg = p8["final_schedule_cfg"]
    ckpt_p = Path(p8["locked_ckpt"]) if Path(p8["locked_ckpt"]).exists() else (REPO / "runs/v34_frontier/stage_c/C2_44m_q32/C2_44m_q32_final.pt")
    pcore, _ = load_pcore_v35(ckpt_p, weight_key=p8["locked_weight"], device=device)
    ecore, raw_stats, priors_ec, priors_a = load_v32_ecore_and_priors(device=device)
    dec = load_s10_decoder(device)

    tasks_all, z_src_all, _ = tasks_and_sources()
    # Pick 30 balanced sources (6 per style * 5 targets = 150 tasks -> 120 cross + 30 same)
    picks = []
    for k in range(0, 150, 5):
        base = k * 5
        picks.extend(range(base, base + 5))
    idx = torch.tensor(picks, dtype=torch.long)
    tasks = [tasks_all[i] for i in picks]

    z_src = z_src_all[idx].to(device).float()
    eps = torch.randn(z_src_all.shape, generator=torch.Generator().manual_seed(EVAL_SEED))[idx].to(device).float()
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    assets = build_or_load_cnx_assets(device=device)
    q16 = assets["stiefel"]["test750"]["q32"][idx].to(device).float()
    cg = assets["stiefel"]["test750"]["cnx_gray"][idx].to(device).float()
    y16 = project_latent_y16_lcn(z_src, v18["luma_proj"]["w"].to(device).float(), v18["luma_proj"]["b"].to(device).float())
    c64 = ts["c_64"][idx].to(device).float()
    tid = torch.tensor([t["target_style_id"] for t in tasks], device=device, dtype=torch.long)
    u_st = match_cnx_k8_submode(cg, tid, assets["bank"])

    def _run_e(lam: float) -> tuple[torch.Tensor, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        zs = []
        for s in range(0, len(tasks), 16):
            sl = slice(s, min(len(tasks), s + 16))
            ze = ecore(
                z_src[sl], c64[sl], q16[sl], tid[sl],
                torch.full((sl.stop - sl.start,), lam, device=device), raw_stats, priors_ec, noise=eps[sl],
            )
            zs.append(build_pcore_startpoint(ze, tid[sl], priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"]))
        torch.cuda.synchronize()
        return torch.cat(zs, 0).cpu(), (time.perf_counter() - t0) * 1000.0 / len(tasks)

    def _run_p(sv: float, tau0_ov: float | None = None, lam_ov: float | None = None) -> tuple[torch.Tensor, float]:
        kw = make_v35_schedule(sv, ws_mul=fcfg["ws_mul"], wc_mul=fcfg["wc_mul"], dtau0=fcfg["dtau0"])
        popped_lam = float(kw.pop("draft_lambda", 0.75))
        lam_d = popped_lam if lam_ov is None else float(lam_ov)
        if tau0_ov is not None:
            kw["tau0"] = tau0_ov
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        zs = []
        for s in range(0, len(tasks), 16):
            sl = slice(s, min(len(tasks), s + 16))
            ze = ecore(
                z_src[sl], c64[sl], q16[sl], tid[sl],
                torch.full((sl.stop - sl.start,), lam_d, device=device), raw_stats, priors_ec, noise=eps[sl],
            )
            ze_s = build_pcore_startpoint(ze, tid[sl], priors_a, alpha_y=fcfg["alpha_y"], alpha_c=fcfg["alpha_c"])
            zo = integrate_v33_trajectory(
                pcore, ze_s, eps[sl], y16[sl], q16[sl], tid[sl], u_st[sl],
                steps=int(fcfg["steps"]), solver="dpm2m", **kw,
            )
            zs.append(zo)
        torch.cuda.synchronize()
        return torch.cat(zs, 0).cpu(), (time.perf_counter() - t0) * 1000.0 / len(tasks)

    # Dense sequence points across H0, H1, H2
    curves: dict[str, list[tuple[str, torch.Tensor, int, float]]] = {"H0_hard_switch": [], "H1_latent_interp": [], "H2_tau_sched_interp": []}

    # H0: E(0.25..0.60) -> hard switch P(0.65..1.00)
    for l_v in (0.25, 0.40, 0.50, 0.60):
        z_pt, ms = _run_e(l_v)
        curves["H0_hard_switch"].append((f"E_{l_v:.2f}", z_pt, 1, ms))
    for s_v in (0.65, 0.70, 0.75, 0.85, 1.00):
        z_pt, ms = _run_p(s_v)
        curves["H0_hard_switch"].append((f"P_{s_v:.2f}", z_pt, 29, ms))

    # H1: Latent blend between E(0.60) and P(0.65)
    z_e60 = curves["H0_hard_switch"][3][1]
    z_p65 = curves["H0_hard_switch"][4][1]
    ms_e60 = curves["H0_hard_switch"][3][3]
    ms_p65 = curves["H0_hard_switch"][4][3]
    for beta in (0.0, 0.25, 0.50, 0.75, 1.0):
        z_b = (1.0 - beta) * z_e60 + beta * z_p65
        nfe = 1 if beta == 0.0 else 29
        ms = ms_e60 if beta == 0.0 else ms_p65
        curves["H1_latent_interp"].append((f"H1_b{int(beta * 100):03d}", z_b, nfe, ms))

    # H2: Continuous schedule tau0 interpolation across boundary: P(s=0.55..0.75) with smooth tau0 -> 0.96..0.785
    for s_v in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        t0_c = round(0.95 - 0.66 * (s_v - 0.50), 4)
        lam_c = round(0.55 + 0.40 * (s_v - 0.50), 3)
        z_pt, ms = _run_p(s_v, tau0_ov=t0_c, lam_ov=lam_c)
        curves["H2_tau_sched_interp"].append((f"H2_s{s_v:.2f}_t{t0_c:.3f}", z_pt, 29, ms))

    # Decode all unique point latents then free generators before referee
    decoded: dict[str, torch.Tensor] = {}
    for c_name, pts in curves.items():
        for lbl, z_t, _, _ in pts:
            key = f"{c_name}::{lbl}"
            decoded[key] = decode_s10(dec, z_t.to(device), batch=16, output_cpu=True).clamp(0.0, 1.0)

    del ecore, pcore, dec
    gc.collect()
    torch.cuda.empty_cache()

    # Measure referee metrics & adjacent RGB LPIPS transitions
    spec_750 = read_json(REPO / "eval_protocol/tasks_750.json")
    src_by_stem = {}
    for srec in spec_750["sources"]:
        with Image.open(REPO / "data/test" / srec["source_rel_path"]) as im:
            arr = np.asarray(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.uint8)
        src_by_stem[srec["source_stem"]] = torch.from_numpy(arr).permute(2, 0, 1)
    src_u8 = torch.stack([src_by_stem[t["source_stem"]] for t in tasks], dim=0)

    src_feats = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt", map_location="cpu", weights_only=False)
    src_dino = np.stack([np.asarray(src_feats["raw_dino"][t["source_stem"]]) for t in tasks], axis=0)
    ea = torch.load(REPO / "runs/v36_art_foundation/cache/v36_eval_assets.pt", map_location="cpu", weights_only=False)
    dec_tmp = load_s10_decoder(device)
    ref_u8 = torch.cat([
        (decode_s10(dec_tmp, ea["refs150"]["z_ref"][s:s + 24].to(device).float(), batch=24, output_cpu=True).clamp(0, 1) * 255).round().to(torch.uint8)
        for s in range(0, len(ea["refs150"]["z_ref"]), 24)
    ], 0)
    del dec_tmp
    gc.collect()
    torch.cuda.empty_cache()

    stack = FeatureStack(device)
    mb = auto_referee_batch(device)
    proto_np = np.asarray(stack.csd_prototypes)
    rf = stack.extract([Image.fromarray(a) for a in ref_u8.permute(0, 2, 3, 1).numpy()], batch=mb)
    ref_dino = [rf["dino"][s * 30:(s + 1) * 30].clone().to(device) for s in range(5)]

    cross_mask = np.array([t["source_style_id"] != t["target_style_id"] for t in tasks])
    tids_np = np.array([t["target_style_id"] for t in tasks])

    out_curves: dict[str, Any] = {}
    for c_name, pts in curves.items():
        pt_rows = []
        adj_rows = []
        prev_z, prev_u8, prev_lbl = None, None, None
        for lbl, z_t, nfe, ms in pts:
            rgb01 = decoded[f"{c_name}::{lbl}"]
            u8 = (rgb01 * 255.0).round().to(torch.uint8)
            fo = stack.extract([Image.fromarray(a) for a in u8.permute(0, 2, 3, 1).numpy()], batch=mb)
            df = fo["dino"].to(device)
            cf = np.asarray(fo["csd"])
            scores = torch.stack([(df @ p.T).max(1).values for p in ref_dino], 1).cpu().numpy()
            cs = cf @ proto_np.T
            dc_arr = (df.cpu().numpy() * src_dino).sum(-1)
            ds_arr = scores[np.arange(len(tasks)), tids_np]
            top1_arr = (cs.argmax(1) == tids_np).astype(np.float32)
            lp_src = stack.lpips_vs(u8, src_u8, batch=mb).cpu().numpy()

            row = {
                "label": lbl,
                "nfe": nfe,
                "lat_ms_per_img_b16": round(ms, 2),
                "dino_c": round(float(dc_arr[cross_mask].mean()), 4),
                "dino_s": round(float(ds_arr[cross_mask].mean()), 4),
                "csd_top1": round(float(top1_arr[cross_mask].mean()), 4),
                "lpips_vs_src": round(float(lp_src[cross_mask].mean()), 4),
            }
            pt_rows.append(row)
            if prev_z is not None:
                l_rmse = float((z_t - prev_z).pow(2).mean().sqrt().item())
                adj_lp = float(stack.lpips_vs(u8, prev_u8, batch=mb).mean().item())
                adj_rows.append({
                    "from": prev_lbl, "to": lbl,
                    "latent_rmse": round(l_rmse, 4),
                    "adjacent_lpips": round(adj_lp, 4),
                    "delta_dino_c": round(row["dino_c"] - pt_rows[-2]["dino_c"], 4),
                    "delta_dino_s": round(row["dino_s"] - pt_rows[-2]["dino_s"], 4),
                })
            prev_z, prev_u8, prev_lbl = z_t, u8, lbl

        out_curves[c_name] = {
            "points": pt_rows,
            "adjacent_jumps": adj_rows,
            "max_latent_rmse_jump": max(x["latent_rmse"] for x in adj_rows) if adj_rows else 0.0,
            "max_adjacent_lpips_jump": max(x["adjacent_lpips"] for x in adj_rows) if adj_rows else 0.0,
        }

    del stack
    gc.collect()
    torch.cuda.empty_cache()

    rep = {"utc": utc(), "n_tasks": len(tasks), "n_cross": int(cross_mask.sum()), "curves": out_curves}
    save_json(out_p, rep)
    print(f"[F5 DONE] saved {out_p}", flush=True)
    return rep


def main() -> None:
    sanitize_cuda_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_f5(a.device)


if __name__ == "__main__":
    main()
