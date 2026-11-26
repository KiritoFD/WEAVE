"""CLEAVE V34 Stage A (0.5h budget, Zero-Training Priority 1):
Reconstructing the Analytic Starting Point by decomposing E-Core draft luminance:
  P_Y z_E = L_low + L_high
Arms (identical noise, Q32, CFG, and frozen V32 P-Core weights):
  - A0: Original E-Core draft z_E
  - A1: Target style low-frequency luminance moment matching (L_high & P_C untouched)
  - A2: Target chrominance P_C distribution matching (P_Y untouched)
  - A3: Joint low-frequency luminance + chrominance redistribution (L_high untouched)
Evaluated at s=0.85_c10 (tau0=0.79) and s=1.00 (tau0=0.635, targeting C ~ 0.45-0.60).
"""
from __future__ import annotations

import argparse
import gc
import math
import time

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v27_single.bare_q32 import load_cache
from exp.v28_unified.student_models import build_student_model
from exp.v29_biglittle.pcore_model import PCoreSLFUNet
from exp.v29_biglittle.phase2_tune import default_knob_schedule
from exp.v32_nodino.build_cnx_assets import build_or_load_cnx_assets
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v34_frontier import (
    OUT,
    S085_C10_KW,
    SEED_V34,
    V32_ECORE_CKPT,
    V32_PCORE_CKPT,
    RawWriterV34,
    compute_style_pixel_stats,
    measure_and_summarize_v34,
    paired_ci_vs_baseline,
)
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import sanitize_cuda_env, save_json, seed_all, utc


def gaussian_lowpass_2d(x: torch.Tensor, kernel_size: int = 19, sigma: float = 4.5) -> torch.Tensor:
    """Separable 2D Gaussian low-pass filter with reflect padding on (B, C, H, W)."""
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=x.device, dtype=torch.float32)
    g1d = torch.exp(-0.5 * (coords / float(sigma)).square())
    g1d = g1d / g1d.sum()
    c = x.shape[1]
    kx = g1d.view(1, 1, 1, kernel_size).repeat(c, 1, 1, 1)
    ky = g1d.view(1, 1, kernel_size, 1).repeat(c, 1, 1, 1)
    xf = x.float()
    h = F.conv2d(F.pad(xf, (half, half, 0, 0), mode="reflect"), kx, groups=c)
    return F.conv2d(F.pad(h, (0, 0, half, half), mode="reflect"), ky, groups=c)


def _py_unit_vec(p_y: torch.Tensor) -> torch.Tensor:
    """Return the unit luminance eigenvector u_y such that P_Y ~= u_y u_y^T."""
    _, _, vh = torch.linalg.svd(p_y.float(), full_matrices=False)
    u = vh[0]
    # Deterministic sign convention: largest magnitude component positive
    idx = int(u.abs().argmax().item())
    if float(u[idx].item()) < 0.0:
        u = -u
    return u


def decompose_py_luma(z: torch.Tensor, p_y: torch.Tensor, *, kernel_size: int = 19, sigma: float = 4.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose P_Y z = L_low + L_high cleanly along rank-1 luminance direction u_y in 4x64x64 latent space."""
    py = p_y.to(device=z.device, dtype=torch.float32)
    zy = torch.einsum("ij,bjhw->bihw", py, z.float())
    l_low = gaussian_lowpass_2d(zy, kernel_size=kernel_size, sigma=sigma)
    l_high = zy - l_low
    return l_low, l_high, zy


def reconstruct_analytic_draft(
    z_e: torch.Tensor,
    tid: torch.Tensor,
    priors_a: dict[str, torch.Tensor],
    *,
    mode: str,
    kernel_size: int = 19,
    sigma: float = 4.5,
) -> torch.Tensor:
    """Apply A0/A1/A2/A3 analytic redistribution to E-Core draft z_E."""
    if mode == "A0":
        return z_e
    dev = z_e.device
    zf = z_e.float()
    p_y = priors_a["P_Y"].to(dev)
    p_c = priors_a["P_C"].to(dev)
    u_y = priors_a["u_y"].to(dev)  # (4,) unit vector

    l_low, l_high, zy = decompose_py_luma(zf, p_y, kernel_size=kernel_size, sigma=sigma)
    zc = torch.einsum("ij,bjhw->bihw", p_c, zf)

    if mode in ("A1", "A3"):
        # Scalar 1D projection along u_y avoids rank-1 negative-component division/sign ambiguity
        s_low = torch.einsum("c,bchw->bhw", u_y, l_low).unsqueeze(1)  # (B, 1, H, W)
        mu_low_t = priors_a["mu_low_1d"].to(dev)[tid]    # (B, 1, 1, 1)
        std_low_t = priors_a["std_low_1d"].to(dev)[tid]  # (B, 1, 1, 1)
        mu_s = s_low.mean(dim=(2, 3), keepdim=True)
        std_s = s_low.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        s_low_new = (s_low - mu_s) / std_s * std_low_t + mu_low_t
        l_low_new = s_low_new * u_y.view(1, 4, 1, 1)
        zy_out = l_low_new + l_high
    else:
        zy_out = zy

    if mode in ("A2", "A3"):
        mu_c_t = priors_a["mu_c"].to(dev)[tid]    # (B, 4, 1, 1)
        std_c_t = priors_a["std_c"].to(dev)[tid]  # (B, 4, 1, 1)
        mu_cs = zc.mean(dim=(2, 3), keepdim=True)
        std_cs = zc.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        zc_new = (zc - mu_cs) / std_cs * std_c_t + mu_c_t
        zc_out = torch.einsum("ij,bjhw->bihw", p_c, zc_new)
    else:
        zc_out = zc

    return zy_out + zc_out


@torch.no_grad()
def build_or_load_analytic_priors(priors_ec: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute target style L_low (1D along u_y) and P_C moments strictly from train_4400 core paintings."""
    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "v34_analytic_priors.pt"
    if p.exists():
        loaded = torch.load(p, map_location="cpu", weights_only=False)
        if "mu_low_1d" in loaded and "u_y" in loaded:
            return loaded

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    cnx_bank = load_cache("runs/v32_nodino/cache/k8_cnx417_bank.pt")
    z_tr = v18["train"]["z"].float()
    core_cls = [ci.cpu().long() for ci in cnx_bank["core_idx_per_style"]]

    p_y = priors_ec["P_Y"].float().cpu()
    p_c = priors_ec["P_C"].float().cpu()
    u_y = _py_unit_vec(p_y)

    mu_low_1d = torch.zeros(5, 1, 1, 1)
    std_low_1d = torch.zeros(5, 1, 1, 1)
    mu_c = torch.zeros(5, 4, 1, 1)
    std_c = torch.zeros(5, 4, 1, 1)

    for s in range(5):
        zs = z_tr[core_cls[s]]
        l_low, _, _ = decompose_py_luma(zs, p_y)
        s_low = torch.einsum("c,bchw->bhw", u_y, l_low).unsqueeze(1)
        zc = torch.einsum("ij,bjhw->bihw", p_c, zs)
        mu_low_1d[s] = s_low.mean().view(1, 1, 1)
        # Target per-image low-freq spatial contrast std (not cross-image variance)
        std_low_1d[s] = s_low.std(dim=(2, 3), unbiased=False).mean().view(1, 1, 1)
        mu_c[s] = zc.mean(dim=(0, 2, 3)).view(4, 1, 1)
        std_c[s] = zc.std(dim=(2, 3), unbiased=False).mean(dim=0).view(4, 1, 1)

    out = {
        "P_Y": p_y,
        "P_C": p_c,
        "u_y": u_y,
        "mu_low_1d": mu_low_1d,
        "std_low_1d": std_low_1d,
        "mu_c": mu_c,
        "std_c": std_c,
    }
    torch.save(out, p)
    return out


@torch.no_grad()
def run_stage_a(device: str = "cuda") -> dict:
    sanitize_cuda_env()
    seed_all(SEED_V34)
    t0 = time.monotonic()

    assets = build_or_load_cnx_assets(device=device)
    stiefel, cnx_bank = assets["stiefel"], assets["bank"]
    tasks, z_src, _ = tasks_and_sources()
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")

    ec_sv = torch.load(V32_ECORE_CKPT, map_location="cpu", weights_only=False)
    raw_stats = ec_sv["raw_stats"].to(device).float()
    priors_ec = {k: v.to(device).float() for k, v in ec_sv["priors"].items()}
    ecore = build_student_model(ec_sv.get("arch", "unet"), raw_stats.shape[-1], device=device).eval().requires_grad_(False)
    ecore.load_state_dict(ec_sv["ema"], strict=True)
    priors_a = build_or_load_analytic_priors(priors_ec)

    pcore = PCoreSLFUNet(base_dim=144, u_dim=417, cond_dim=320, use_gamma=True, normalize_u=True).to(device).eval().requires_grad_(False)
    pcore.load_state_dict(torch.load(V32_PCORE_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)

    dec = load_s10_decoder(device)
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks], dtype=torch.long)
    y16_750 = project_latent_y16_lcn(z_src.float(), v18["luma_proj"]["w"].float(), v18["luma_proj"]["b"].float())
    c64_750 = ts["c_64"].float()
    q32_750 = stiefel["test750"]["q32"].float()
    cnx_gray_750 = stiefel["test750"]["cnx_gray"].float()
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    # Precompute u_style & E-Core drafts for the two working points (s=0.85_c10 & s=1.00)
    u_750_list = []
    for st in range(0, len(tasks), 128):
        sl = slice(st, min(st + 128, len(tasks)))
        u_750_list.append(match_cnx_k8_submode(cnx_gray_750[sl].to(device), tid_750[sl].to(device), cnx_bank).cpu())
    u_750 = torch.cat(u_750_list, 0)

    knots = [
        ("s085", S085_C10_KW),
        ("s100", default_knob_schedule(1.00, use_gamma=True)),
    ]
    ze_cache: dict[str, torch.Tensor] = {}
    for k_tag, kw in knots:
        lam = float(kw["draft_lambda"])
        chunks = []
        for st in range(0, len(tasks), 32):
            sl = slice(st, min(st + 32, len(tasks)))
            chunks.append(ecore(
                z_src[sl].to(device).float(), c64_750[sl].to(device).float(),
                q32_750[sl].to(device).float(), tid_750[sl].to(device),
                torch.full((sl.stop - sl.start,), lam, device=device),
                raw_stats, priors_ec, noise=eps_750[sl].to(device).float(),
            ).cpu())
        ze_cache[k_tag] = torch.cat(chunks, 0)

    arms_out = {}
    for k_tag, kw in knots:
        kw_call = {k: v for k, v in kw.items() if k != "draft_lambda"}
        ze_orig = ze_cache[k_tag]
        for mode in ("A0", "A1", "A2", "A3"):
            arm_name = f"StageA_{mode}_{k_tag}"
            m_dir = OUT / "measured" / arm_name
            if not (m_dir / "summary_v34.json").exists():
                writer = RawWriterV34(arm_name, dec=dec, device=device)
                for st in range(0, len(tasks), 32):
                    sl = slice(st, min(st + 32, len(tasks)))
                    ze_mod = reconstruct_analytic_draft(ze_orig[sl].to(device), tid_750[sl].to(device), priors_a, mode=mode)
                    zo = integrate_v33_trajectory(
                        pcore, ze_mod, eps_750[sl].to(device).float(),
                        y16_750[sl].to(device).float(), q32_750[sl].to(device).float(),
                        tid_750[sl].to(device), u_750[sl].to(device).float(),
                        steps=6, solver="dpm2m", **kw_call,
                    )
                    writer.write_batch(zo, tasks[sl])
                writer.finalize(arm=arm_name, mode=mode, knot=k_tag, schedule=kw)
            res = measure_and_summarize_v34(arm_name, device=device)
            pix = compute_style_pixel_stats(OUT / "raw" / arm_name, target_style_id=2, cross_only=True)
            res["min_pixel_stats"] = pix
            arms_out[arm_name] = res
            print(
                f"[Stage A] {arm_name:18s} | C={res['dino_c']:.4f} S={res['dino_s']:.4f} Top1={res['csd_top1']*100:5.2f}% "
                f"| Min(C={res['per_tgt_dino_c']['Min']:.4f}, S={res['per_tgt_dino_s']['Min']:.4f}, sat={pix['sat_hsv']:.4f}, gstd={pix['gray_std']:.4f}) "
                f"| S_tgt:{res['per_tgt_dino_s']}",
                flush=True,
            )

    cis = {}
    for k_tag, _ in knots:
        for mode in ("A1", "A2", "A3"):
            cis[f"{mode}_vs_A0_{k_tag}"] = paired_ci_vs_baseline(
                OUT / f"measured/StageA_{mode}_{k_tag}/eval.json",
                OUT / f"measured/StageA_A0_{k_tag}/eval.json",
            )

    # Determine whether Stage A achieved an independent Minimalism / overall gain
    best_s100_mode = max(("A1", "A2", "A3"), key=lambda m: arms_out[f"StageA_{m}_s100"]["per_tgt_dino_s"]["Min"])
    min_s_gain = arms_out[f"StageA_{best_s100_mode}_s100"]["per_tgt_dino_s"]["Min"] - arms_out["StageA_A0_s100"]["per_tgt_dino_s"]["Min"]
    min_sat_gain = arms_out[f"StageA_{best_s100_mode}_s100"]["min_pixel_stats"]["sat_hsv"] - arms_out["StageA_A0_s100"]["min_pixel_stats"]["sat_hsv"]
    overall_ds = arms_out[f"StageA_{best_s100_mode}_s100"]["dino_s"] - arms_out["StageA_A0_s100"]["dino_s"]
    stage_a_effective = bool((min_sat_gain >= 0.015 or min_s_gain >= 0.015) and overall_ds >= -0.005)

    del ecore, pcore, dec
    gc.collect()
    torch.cuda.empty_cache()

    rep = {
        "updated_utc": utc(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "arms": arms_out,
        "cis": cis,
        "best_s100_min_mode": best_s100_mode,
        "min_dino_s_gain_s100": round(min_s_gain, 4),
        "min_sat_gain_s100": round(min_sat_gain, 4),
        "overall_dino_s_delta_s100": round(overall_ds, 4),
        "stage_a_effective": stage_a_effective,
    }
    save_json(OUT / "stage_a_report.json", rep)
    print(f"[Stage A DONE] in {rep['elapsed_s']:.1f}s | effective={stage_a_effective} (best={best_s100_mode}, dMinS={min_s_gain:+.4f}, dSat={min_sat_gain:+.4f})", flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="CLEAVE V34 Stage A Analytic Startpoint")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_stage_a(device=args.device)


if __name__ == "__main__":
    main()
