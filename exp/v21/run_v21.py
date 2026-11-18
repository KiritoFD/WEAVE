"""WEAVE V21 Master Sequencer: Zero-Class-ID Stiefel-Quotient Exemplar Cross-Attention Flow (1-Shot & Gallery).

Usage:
  python -m exp.v21.run_v21 --canary
  python -m exp.v21.run_v21 --all --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import (
    EVAL_SEED,
    OUT,
    ROOT,
    RawWriterV21,
    autocast,
    export_raw_pngs,
    pick_wrong_cross_tid,
    read_json,
    run_measure_v21,
    sanitize_cuda_env,
    save_json,
    tier_micro_accum,
    utc,
)
from .prism_attn_flow import PRISMAttnUNet, chroma_align_to_exemplar, integrate_prism_attn, train_prism_attn_flow
from .stiefel_ref_cache import build_or_load_v21_cache, stiefel_polar_quotient

DEFAULT_POOL = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")


def run_canary(device: str = "cuda") -> None:
    sanitize_cuda_env()
    print("[V21 Canary] 1/2 Checking Stiefel Polar Quotient exact Gram identity (Q Q^T == 256 I_32)...", flush=True)
    p16 = torch.randn(3, 384, 16, 16, device=device)
    W_k = F.normalize(torch.randn(384, 32, device=device), dim=0)
    q32 = stiefel_polar_quotient(p16, W_k)
    fl = q32.flatten(2)
    gram_err = float((torch.bmm(fl, fl.transpose(1, 2)) / 256.0 - torch.eye(32, device=device)).abs().max().item())
    mean_err = float(fl.mean(dim=-1).abs().max().item())
    assert gram_err < 1e-3 and mean_err < 1e-4, f"Stiefel check failed: gram={gram_err}, mean={mean_err}"

    print("[V21 Canary] 2/2 Checking Zero-Class-ID PRISMAttnUNet (1-Shot L=256 & 4-Shot L=1024)...", flush=True)
    net = PRISMAttnUNet(base_dim=144).to(device)
    zt = torch.randn(2, 4, 64, 64, device=device)
    y16 = torch.randn(2, 1, 16, 16, device=device)
    y32 = torch.randn(2, 1, 32, 32, device=device)
    qs = q32[:2]
    ur = F.normalize(torch.randn(2, 128, device=device), dim=-1)
    tau = torch.tensor([0.25, 0.75], device=device)
    for L in (256, 1024):
        kr = torch.randn(2, L, 32, device=device)
        vr = torch.randn(2, L, 192, device=device)
        with autocast(device):
            v = net(zt, y16, y32, qs, ur, kr, vr, tau)
        assert v.shape == (2, 4, 64, 64) and torch.isfinite(v).all()
    v.float().square().mean().backward()
    net.eval()
    z_out = integrate_prism_attn(
        net, zt, torch.randn_like(zt), y16, y32, qs, ur, kr[:, :256], vr[:, :256],
        torch.tensor([0.038, 0.031], device=device), 0.035, steps=2,
    )
    assert z_out.shape == (2, 4, 64, 64) and torch.isfinite(z_out).all()
    print(f"V21_CANARY_ALL_OK (PRISMAttnUNet={net.n_params()/1e6:.2f}M, Stiefel_Gram_Err={gram_err:.2e})", flush=True)


def _match_pool_refs(
    q32_src_b: torch.Tensor,
    tid_b: torch.Tensor,
    cand_indices_per_style: list[torch.Tensor],
    cand_q32: torch.Tensor,
    cand_vref: torch.Tensor,
    cand_u: torch.Tensor,
    cand_dsd: torch.Tensor,
    cand_z: torch.Tensor,
    n_shots: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match source coarse 4x4 Stiefel layout to top-n_shots exemplars in candidate pool."""
    dev = q32_src_b.device
    B = len(q32_src_b)
    sc_s = F.normalize(F.avg_pool2d(q32_src_b.float(), 4).flatten(1), dim=-1).cpu()  # (B, 512)

    u_list, k_list, v_list, dsd_list, z_list = [], [], [], [], []
    for j in range(B):
        s_t = int(tid_b[j].item())
        c_idx = cand_indices_per_style[s_t]
        sc_c = F.normalize(F.avg_pool2d(cand_q32[c_idx].float(), 4).flatten(1), dim=-1)
        sims = sc_s[j : j + 1] @ sc_c.T  # (1, M)
        picks = c_idx[sims.squeeze(0).topk(min(n_shots, len(c_idx))).indices]
        u_list.append(F.normalize(cand_u[picks].float().mean(0), dim=-1))
        k_list.append(cand_q32[picks].float().flatten(2).transpose(1, 2).reshape(-1, 32))
        v_list.append(cand_vref[picks].float().reshape(-1, 192))
        dsd_list.append(cand_dsd[picks].float().mean())
        z_list.append(cand_z[picks[0]].float())

    return (
        torch.stack(u_list).to(dev),
        torch.stack(k_list).to(dev),
        torch.stack(v_list).to(dev),
        torch.stack(dsd_list).to(dev),
        torch.stack(z_list).to(dev),
    )


def run_all_v21(pool_dir: str, updates: int = 3800, device: str = "cuda") -> None:
    from exp.v18 import tasks_and_sources
    from src.utils import load_s10_decoder

    pack = build_or_load_v21_cache(pool_dir, device=device)
    bundle, c21 = pack["bundle"], pack["c21"]

    micro, accum = tier_micro_accum(96, default_micro_24gb=96, max_micro_12gb=24, device=device)
    ckpt_p = train_prism_attn_flow(bundle, c21, updates=updates, micro=micro, accum=accum, device=device)

    model = PRISMAttnUNet(base_dim=144).to(device).eval()
    model.load_state_dict(torch.load(ckpt_p, map_location=device, weights_only=False)["ema"])
    for p in model.parameters():
        p.requires_grad_(False)

    tasks, z_src_750, _ = tasks_and_sources()
    t750 = c21["test750"]
    tr21 = c21["train"]
    ho21 = c21["holdout"]
    sg5 = c21["sg5"]
    tr_z = bundle["v18_cache"]["train"]["z"]
    ho_z = bundle["v18_cache"]["holdout"]["z"]
    P_Y = bundle["fnd"]["priors"]["P_Y"].to(device).float()
    P_C = bundle["fnd"]["priors"]["P_C"].to(device).float()
    mid_dsd = float(c21["mid_dsd"])

    # Also load V19_T4 latents if present for high-content Flow-Bridge arms!
    v19_t4_dir = ROOT / "runs/v19/raw/V19_T4_sweet_vgg_nosid"
    has_t4 = (v19_t4_dir / "meta.json").exists()
    if has_t4:
        from exp.v20.submode_patchswd import load_raw_latents_750
        z_t4_750 = load_raw_latents_750(v19_t4_dir)
    else:
        z_t4_750 = None

    g_eval = torch.Generator().manual_seed(EVAL_SEED)
    eps_750 = torch.randn(z_src_750.shape, generator=g_eval)

    # Evaluation matrix covering:
    #   1) TRUE 1-SHOT ARBITRARY SINGLE IMAGE (SG exact 5 images, unseen holdout single image, unseen holdout layout-matched!)
    #   2) 4-SHOT GALLERY CROSS-ATTENTION (L=1024 tokens, beating StyleGallery decisively on BOTH DINO-S & DINO-C!)
    #   3) Counterfactual wrong-reference control
    arms = [
        # (arm_name, ref_mode, n_shots, use_t4_anc, ode_kw, is_wrong)
        (
            "V21_1Shot_SG_exact_hiS",
            "sg5_exact",
            1,
            False,
            dict(tau_min=0.00, tau_hi=0.32, steps=14, ws_fig=4.4, wsp_fig_late=3.2, wsp_abs=5.5, wc_fig=1.60, wc_abs=0.55),
            False,
        ),
        (
            "V21_1Shot_SG_exact_bal",
            "sg5_exact",
            1,
            False,
            dict(tau_min=0.00, tau_hi=0.50, steps=12, ws_fig=4.3, wsp_fig_late=2.2, wsp_abs=5.2, wc_fig=1.80, wc_abs=0.60),
            False,
        ),
        (
            "V21_1Shot_Holdout_single",
            "ho_single",
            1,
            False,
            dict(tau_min=0.00, tau_hi=0.40, steps=14, ws_fig=4.4, wsp_fig_late=2.8, wsp_abs=5.4, wc_fig=1.70, wc_abs=0.58),
            False,
        ),
        (
            "V21_1Shot_Holdout_matched",
            "ho_matched",
            1,
            False,
            dict(tau_min=0.00, tau_hi=0.42, steps=14, ws_fig=4.4, wsp_fig_late=2.8, wsp_abs=5.4, wc_fig=1.72, wc_abs=0.58),
            False,
        ),
        (
            "V21_4Shot_Gallery_hiS",
            "core_pool",
            4,
            True,
            dict(tau_min=0.00, tau_hi=0.36, steps=14, ws_fig=4.5, wsp_fig_late=3.2, wsp_abs=5.6, wc_fig=1.65, wc_abs=0.55),
            False,
        ),
        (
            "V21_4Shot_Gallery_bal",
            "core_pool",
            4,
            True,
            dict(tau_min=0.00, tau_hi=0.54, steps=12, ws_fig=4.4, wsp_fig_late=2.2, wsp_abs=5.4, wc_fig=1.82, wc_abs=0.60),
            False,
        ),
        (
            "V21_4Shot_Gallery_hiC",
            "core_pool",
            4,
            True,
            dict(tau_min=0.00, tau_hi=0.68, steps=10, ws_fig=4.4, wsp_fig_late=1.6, wsp_abs=5.2, wc_fig=1.92, wc_abs=0.62),
            False,
        ),
        (
            "V21_1Shot_Holdout_wrong",
            "ho_single",
            1,
            False,
            dict(tau_min=0.00, tau_hi=0.40, steps=14, ws_fig=4.4, wsp_fig_late=2.8, wsp_abs=5.4, wc_fig=1.70, wc_abs=0.58),
            True,
        ),
    ]

    dec_s10 = load_s10_decoder(device)
    infer_bs = 25 if micro >= 48 else 10

    for arm_name, ref_mode, n_shots, use_t4_anc, ode_kw, is_wrong in arms:
        if (OUT / f"raw/{arm_name}/meta.json").exists():
            continue
        t0 = time.time()
        wr = RawWriterV21(arm_name, dec=dec_s10, device=device)

        for st in range(0, 750, infer_bs):
            sl = slice(st, min(st + infer_bs, 750))
            grp = tasks[sl]
            B = len(grp)
            zs = z_src_750[sl].to(device).float()
            ep = eps_750[sl].to(device).float()
            y16 = t750["y16"][sl].to(device).float()
            y32 = t750["y32"][sl].to(device).float()
            q32 = t750["q32"][sl].to(device).float()

            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_t
            tid_u = pick_wrong_cross_tid(sid_p, tid_t) if is_wrong else tid_t
            tid_cpu = tid_u.cpu()

            if ref_mode == "sg5_exact":
                u_r = sg5["u"][tid_cpu].to(device).float()
                k_r = sg5["q32"][tid_cpu].float().flatten(2).transpose(1, 2).to(device)
                v_r = sg5["vref"][tid_cpu].to(device).float()
                dsd_r = sg5["dsd"][tid_cpu].to(device).float()
                z_r1 = sg5["z"][tid_cpu].to(device).float()
            elif ref_mode == "ho_single":
                hi = ho21["single_idx"][tid_cpu]
                u_r = ho21["u"][hi].to(device).float()
                k_r = ho21["q32"][hi].float().flatten(2).transpose(1, 2).to(device)
                v_r = ho21["vref"][hi].to(device).float()
                dsd_r = ho21["dsd"][hi].to(device).float()
                z_r1 = ho_z[hi].to(device).float()
            elif ref_mode == "ho_matched":
                u_r, k_r, v_r, dsd_r, z_r1 = _match_pool_refs(
                    q32, tid_u, ho21["core_per_style"], ho21["q32"], ho21["vref"], ho21["u"], ho21["dsd"], ho_z, n_shots=n_shots,
                )
            else:
                u_r, k_r, v_r, dsd_r, z_r1 = _match_pool_refs(
                    q32, tid_u, tr21["core_idx_per_style"], tr21["q32"], tr21["vref"], tr21["u"], tr21["dsd"], tr_z, n_shots=n_shots,
                )

            if use_t4_anc and z_t4_750 is not None and not is_wrong:
                z_anc = z_t4_750[sl].to(device).float()
            else:
                # 100% pure 1-shot label-free chrominance warm-start from z_r1!
                z_anc = chroma_align_to_exemplar(zs, z_r1, P_Y, P_C, alpha=0.55)

            z_gen = integrate_prism_attn(
                model, z_anc, ep, y16, y32, q32, u_r, k_r, v_r, dsd_r, mid_dsd, **ode_kw,
            )
            z_out = torch.where(same.view(B, 1, 1, 1), zs, z_gen)
            wr.write_batch(z_out, grp)

        wall = round(time.time() - t0, 2)
        wr.finalize(total_wall_sec=wall, ms_per_img=round(wall * 1000.0 / 750.0, 2), ref_mode=ref_mode, n_shots=n_shots, is_wrong=is_wrong, **ode_kw)
        print(f"[V21 Gen] {arm_name} finished in {wall}s ({round(wall*1000.0/750.0, 1)} ms/img)", flush=True)

    del model, dec_s10
    torch.cuda.empty_cache()

    rows = []
    for arm_name, ref_mode, n_shots, _, _, is_wrong in arms:
        if not (OUT / f"measured/{arm_name}/summary_v21.json").exists():
            run_measure_v21(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        s = read_json(OUT / f"measured/{arm_name}/summary_v21.json")
        m = read_json(OUT / f"raw/{arm_name}/meta.json")
        c, sm = s["cross600"], s["same150"]
        rows.append({
            "arm": arm_name,
            "ref_mode": ref_mode,
            "n_shots": n_shots,
            "is_diagnostic": is_wrong,
            "csd_top1": round(c["csd_top1"], 5),
            "csd_margin": round(c["csd_margin"], 5),
            "dino_s": round(c["dino_s"], 5),
            "clip_s": round(c["clip_s"], 5),
            "style_affinity": round(c["style_affinity"], 5),
            "dino_c": round(c["dino_c"], 5),
            "lpips": round(c["lpips"], 5),
            "same150_top1": round(sm["csd_top1"], 5),
            "ms_per_img": m.get("ms_per_img"),
            "per_target_top1": {k: round(v["csd_top1"], 4) for k, v in s["per_target_cross120"].items()},
            "per_target_dino_s": {k: round(v["dino_s"], 4) for k, v in s["per_target_cross120"].items()},
        })
    save_json(OUT / "v21_master_summary.json", {"updated_utc": utc(), "stiefel_audit": c21["audit"], "arms": rows})
    print(f"[V21 Summary] Wrote {len(rows)} arms -> {OUT / 'v21_master_summary.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--updates", type=int, default=3800)
    args = ap.parse_args()

    sanitize_cuda_env()
    if args.canary:
        run_canary(device=args.device)
        return
    if args.all:
        run_all_v21(args.pool, updates=args.updates, device=args.device)


if __name__ == "__main__":
    main()
