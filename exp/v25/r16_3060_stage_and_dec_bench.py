"""V25 R16 (3060): RTX 3060 12GB 全栈端到端阶段拆分 + PRISM-Flow (D2) + Full SD-VAE vs S10 解码测速.

保证论文中所有延迟/吞吐/阶段耗时数据 100% 为同一张 RTX 3060 12GB 实测口径:
  1. 阶段拆分 (batch=1 & batch=16):
     - prism1, d6_csj_b4, d6_full_b4, prism_hc_r12, prism_flow_d2, o3_ref_fresh, o3_ref_cached
     - 拆分 encode (A0) / feature (DINO/CNX/LCN/27D) / style / anchor / flow / decode (S10) / total_rgb2rgb / cached_flow_dec
  2. prism_flow_d2 (V20_D2_core_slf_bal, 16-step Core-SLF) 在 3060 12GB 下扫 batch ∈ {1, 4, 8, 16, 24, 32, 48, 64}
  3. Decoder 对照 (Full SD-VAE 49.5M fp16 vs PRISM-Dec S10 10.66M fp16) @ batch ∈ {1, 8, 16}

用法 (3060):
  python -m exp.v25.r16_3060_stage_and_dec_bench --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn.functional as F

from . import OUT, ROOT, sanitize_cuda_env, save_json, seed_all, utc
from exp.v18.structure import extract_whitened_dino_layout, rgb_to_structure_64
from exp.v19.fiber_duality import (
    StyleFiberEncoder,
    apply_latent_chroma_shift,
    build_or_load_v19_foundation,
    extract_color_fiber_27d,
    extract_dino_rgb_moments_768d,
)
from exp.v19.routed_student import SpatiallyRoutedChordNet
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, integrate_slf_core_batch
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline, match_scene_submode
from exp.v22.core import CFGConfig, D6_SOLVER, GateConfig, build_content_condition, effective_gate_components, make_initial_state, predict_guided_velocity, rk2_integrate
from exp.v22.ref_adapter import RefAdapter
from exp.v24.p0_m0_m2 import P1_CKPT, V20_CKPT
from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot
from exp.v25.precompute_train_latents import B4_CKPT
from exp.v25.r1_inference_ablation import D6H_SOLVER
from exp.v25.r14_e2e_benchmark import _SD_VAE_CFG, _summarize, load_a0_encoder
from src.infer import canonicalizer, load_c3unet
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import load_dino_model

OUT_JSON = OUT / "r16_3060_stage_and_dec_bench.json"
O3_CKPT = ROOT / "runs/v24/oneshot_val/O3_rgb_cnx_gated.pt"


def _sync(device: str):
    torch.cuda.synchronize(device)


@torch.no_grad()
def bench_decoders(device: str = "cuda") -> dict:
    from diffusers import AutoencoderKL

    s10 = load_s10_decoder(device)
    full_vae = AutoencoderKL.from_config(_SD_VAE_CFG).to(device).half().eval()
    for p in full_vae.parameters():
        p.requires_grad_(False)

    res = {}
    for bs in (1, 8):
        z = torch.randn(bs, 4, 64, 64, device=device)
        # Warmup
        for _ in range(5):
            _ = decode_s10(s10, z, batch=bs, output_cpu=False)
            _ = full_vae.decode(z.half() / 0.18215).sample
        _sync(device)

        s10_times, full_times = [], []
        for _ in range(25):
            _sync(device); t0 = time.perf_counter()
            _ = decode_s10(s10, z, batch=bs, output_cpu=False)
            _sync(device); s10_times.append((time.perf_counter() - t0) * 1000.0)

            _sync(device); t0 = time.perf_counter()
            _ = full_vae.decode(z.half() / 0.18215).sample
            _sync(device); full_times.append((time.perf_counter() - t0) * 1000.0)

        s10_p50_batch = float(np.percentile(s10_times, 50))
        full_p50_batch = float(np.percentile(full_times, 50))
        s10_mean_img = float(np.mean(s10_times)) / bs
        full_mean_img = float(np.mean(full_times)) / bs
        res[f"b{bs}"] = {
            "batch": bs,
            "s10_p50_batch_ms": round(s10_p50_batch, 2),
            "full_p50_batch_ms": round(full_p50_batch, 2),
            "s10_p50_ms_per_img": round(s10_p50_batch / bs, 2),
            "full_p50_ms_per_img": round(full_p50_batch / bs, 2),
            "s10_mean_ms_per_img": round(s10_mean_img, 2),
            "full_mean_ms_per_img": round(full_mean_img, 2),
            "speedup_p50": round(full_p50_batch / s10_p50_batch, 2),
        }
        print(f"[R16 DEC b={bs}] Full={full_p50_batch:.2f}ms ({full_p50_batch/bs:.2f}/img) | "
              f"S10={s10_p50_batch:.2f}ms ({s10_p50_batch/bs:.2f}/img) | speedup={full_p50_batch/s10_p50_batch:.2f}x", flush=True)

    del s10, full_vae
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def bench_stages_3060(bundle, bank, device: str = "cuda") -> dict:
    vae = load_a0_encoder(device)
    dec = load_s10_decoder(device)
    dino = load_dino_model(device=device)
    cnx = load_convnext_tiny_offline(device).eval()
    cnx_s2 = cnx[:4]
    b4, _ = load_c3unet(B4_CKPT, device)
    canon = canonicalizer(device)

    fnd = bundle["fnd"]
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    bary = fiber_enc.class_barycenters().detach()
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)

    p1 = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    p1.load_state_dict(torch.load(P1_CKPT, map_location="cpu", weights_only=False)["model"])

    slf = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
    slf.load_state_dict(torch.load(V20_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)

    ck_o3 = torch.load(O3_CKPT, map_location="cpu", weights_only=False)
    o3_fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device).eval()
    o3_fm.load_state_dict(ck_o3["fm"], strict=True)
    o3_ad = RefAdapter(dim=128).to(device).eval()
    o3_ad.load_state_dict(ck_o3["adapter"], strict=True)
    o3_fib_m, o3_fib_s = ck_o3["fib_mean"].to(device), ck_o3["fib_std"].to(device)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    mid_dsd = float(bank["mid_dsd"])

    def _encode_chunked(rgb_in: torch.Tensor, sub: int = 8) -> torch.Tensor:
        outs = []
        for s in range(0, len(rgb_in), sub):
            x = (rgb_in[s:s + sub].half() * 2.0 - 1.0)
            outs.append((vae.encode(x).latent_dist.mean * 0.18215).float())
        return torch.cat(outs, 0)

    def _run_batch(engine_name: str, B: int) -> dict:
        rgb_in = torch.rand(B, 3, 512, 512, device=device)
        rgb_ref = torch.rand(1, 3, 512, 512, device=device).expand(B, -1, -1, -1)
        tid = torch.full((B,), 2, dtype=torch.long, device=device)
        eps = torch.randn(B, 4, 64, 64, device=device)
        timings = {}

        if engine_name == "o3_ref_cached":
            d16_r = extract_whitened_dino_layout(dino, rgb_ref[:1], batch=1, device=device).expand(B, -1, -1, -1)
            x_r = (F.interpolate(rgb_ref[:1], size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
            v16_r = F.avg_pool2d(cnx_s2(x_r), 2, 2)
            v16_r = (v16_r / v16_r.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)).expand(B, -1, -1, -1)
            fib_r = torch.cat([extract_color_fiber_27d(rgb_ref[:1]),
                               extract_dino_rgb_moments_768d(dino, rgb_ref[:1], batch=1, device=device).to(device)], dim=-1)
            u_r_enc = fiber_enc.encode(fib_r.float()).expand(B, -1)
            u_glob = ((fib_r - o3_fib_m) / o3_fib_s).expand(B, -1)

        # Stage 1: Encode
        _sync(device); t0 = time.perf_counter()
        z_src = _encode_chunked(rgb_in, sub=8)
        _sync(device); timings["encode"] = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Feature + Style + Anchor + Flow
        if engine_name == "prism1":
            _sync(device); tf0 = time.perf_counter()
            d16 = extract_whitened_dino_layout(dino, rgb_in, batch=min(B, 16), device=device)
            col27 = extract_color_fiber_27d(rgb_in)
            d768 = extract_dino_rgb_moments_768d(dino, rgb_in, batch=min(B, 16), device=device).to(device)
            fib = torch.cat([col27, d768], dim=-1)
            c64 = rgb_to_structure_64(rgb_in)
            _sync(device); timings["feature"] = (time.perf_counter() - tf0) * 1000.0

            _sync(device); ts0 = time.perf_counter()
            chord = bary[tid] - fiber_enc.encode(fib.float())
            _sync(device); timings["style"] = (time.perf_counter() - ts0) * 1000.0

            _sync(device); ta0 = time.perf_counter()
            z_anc = apply_latent_chroma_shift(z_src, tid, priors, alpha=0.20)
            z_luma = torch.einsum("ij,bjhw->bihw", P_Y, z_src)
            _sync(device); timings["anchor"] = (time.perf_counter() - ta0) * 1000.0

            _sync(device); tfl0 = time.perf_counter()
            z_out = z_anc + p1(z_anc, z_luma, c64, d16, chord)
            _sync(device); timings["flow"] = (time.perf_counter() - tfl0) * 1000.0

        elif engine_name in ("d6_csj_b4", "d6_full_b4", "prism_hc_r12"):
            is_hc = (engine_name == "prism_hc_r12")
            branches = "full" if engine_name == "d6_full_b4" else "cs_j"
            solver = dict(D6_SOLVER if engine_name == "d6_full_b4" else D6H_SOLVER)

            _sync(device); tf0 = time.perf_counter()
            d16 = extract_whitened_dino_layout(dino, rgb_in, batch=min(B, 16), device=device)
            luma = ((z_src * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)
            y16 = coarse_y16_lcn(luma)
            if is_hc:
                col27 = extract_color_fiber_27d(rgb_in)
                d768 = extract_dino_rgb_moments_768d(dino, rgb_in, batch=min(B, 16), device=device).to(device)
                fib = torch.cat([col27, d768], dim=-1)
                c64 = rgb_to_structure_64(rgb_in)
            _sync(device); timings["feature"] = (time.perf_counter() - tf0) * 1000.0

            _sync(device); ts0 = time.perf_counter()
            _, u_style = match_scene_submode(d16, tid, bank)
            sc_src = F.normalize(F.avg_pool2d(d16.float(), 4).flatten(1), dim=-1)
            best_c = torch.einsum("bd,bkd->bk", sc_src, bank["sub_scene"].to(device).float()[tid]).argmax(dim=1)
            dsd_ref = bank["sub_dsd"].to(device).float()[tid, best_c]
            if is_hc:
                chord = bary[tid] - fiber_enc.encode(fib.float())
            _sync(device); timings["style"] = (time.perf_counter() - ts0) * 1000.0

            _sync(device); ta0 = time.perf_counter()
            c_can = canon(z_src, mode="c1_gentle", augment=False)
            z_anc_b4 = b4(c_can, 0.0, c_can, tid)
            if is_hc:
                z_anc_p1 = apply_latent_chroma_shift(z_src, tid, priors, alpha=0.20)
                z_luma = torch.einsum("ij,bjhw->bihw", P_Y, z_src)
                z_p1 = z_anc_p1 + p1(z_anc_p1, z_luma, c64, d16, chord)
            _sync(device); timings["anchor"] = (time.perf_counter() - ta0) * 1000.0

            _sync(device); tfl0 = time.perf_counter()
            g_tau, g_layout, g_guid, _ = effective_gate_components(dsd_ref, GateConfig("original"), mid_dsd)
            lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
            content = build_content_condition(y16.float(), d16.float(), g_layout)
            tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
            z = make_initial_state(eps, z_anc_b4, tau_v)
            cfg = CFGConfig(branches=branches, rescale_phi=0.48)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vel_fn = lambda zz, prog: predict_guided_velocity(
                    slf, zz, tau_v + prog * (1.0 - tau_v), content, tid, u_style, lscale, g_guid, cfg, solver, prog,
                )
                z_d6 = rk2_integrate(vel_fn, z, tau_v, int(solver["steps"]))
            z_out = (0.45 * z_p1 + 0.55 * z_d6) if is_hc else z_d6
            _sync(device); timings["flow"] = (time.perf_counter() - tfl0) * 1000.0

        elif engine_name == "prism_flow_d2":
            _sync(device); tf0 = time.perf_counter()
            d16 = extract_whitened_dino_layout(dino, rgb_in, batch=min(B, 16), device=device)
            luma = ((z_src * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)
            y16 = coarse_y16_lcn(luma)
            _sync(device); timings["feature"] = (time.perf_counter() - tf0) * 1000.0

            _sync(device); ts0 = time.perf_counter()
            _, _ = match_scene_submode(d16, tid, bank)
            _sync(device); timings["style"] = (time.perf_counter() - ts0) * 1000.0

            _sync(device); ta0 = time.perf_counter()
            c_can = canon(z_src, mode="c1_gentle", augment=False)
            z_anc_b4 = b4(c_can, 0.0, c_can, tid)
            _sync(device); timings["anchor"] = (time.perf_counter() - ta0) * 1000.0

            _sync(device); tfl0 = time.perf_counter()
            z_out = integrate_slf_core_batch(
                slf, z_anc_b4, eps, y16.float(), d16.float(), tid, bank,
                tau_hi=0.44, steps=16, ws_fig=3.8, wsp_fig_late=4.8, wsp_abs=5.4,
                wc_fig=1.60, wc_abs=0.60, lscale_abs=0.28,
            )
            _sync(device); timings["flow"] = (time.perf_counter() - tfl0) * 1000.0

        elif engine_name in ("o3_ref_fresh", "o3_ref_cached"):
            cached_ref = (engine_name == "o3_ref_cached")
            _sync(device); tf0 = time.perf_counter()
            y16 = coarse_y16_lcn(((z_src * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0))
            d16 = extract_whitened_dino_layout(dino, rgb_in, batch=min(B, 16), device=device)
            fib_c = torch.cat([extract_color_fiber_27d(rgb_in),
                               extract_dino_rgb_moments_768d(dino, rgb_in, batch=min(B, 16), device=device).to(device)], dim=-1)
            if not cached_ref:
                d16_r = extract_whitened_dino_layout(dino, rgb_ref, batch=min(B, 16), device=device)
                x_r = (F.interpolate(rgb_ref, size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
                v16_r = F.avg_pool2d(cnx_s2(x_r), 2, 2)
                v16_r = v16_r / v16_r.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
                fib_r = torch.cat([extract_color_fiber_27d(rgb_ref),
                                   extract_dino_rgb_moments_768d(dino, rgb_ref, batch=min(B, 16), device=device).to(device)], dim=-1)
            _sync(device); timings["feature"] = (time.perf_counter() - tf0) * 1000.0

            _sync(device); ts0 = time.perf_counter()
            u_c_enc = fiber_enc.encode(fib_c.float())
            if not cached_ref:
                u_r_enc = fiber_enc.encode(fib_r.float())
                u_glob = (fib_r - o3_fib_m) / o3_fib_s
            u_style = o3_ad(u_r_enc, u_c_enc)
            _sync(device); timings["style"] = (time.perf_counter() - ts0) * 1000.0

            _sync(device); ta0 = time.perf_counter()
            tau0, wj, steps = 0.45, 2.2, 10
            z = (1.0 - tau0) * eps + tau0 * z_src
            _sync(device); timings["anchor"] = (time.perf_counter() - ta0) * 1000.0

            _sync(device); tfl0 = time.perf_counter()
            f0 = torch.zeros(B, dtype=torch.bool, device=device)
            f1 = torch.ones(B, dtype=torch.bool, device=device)
            l1 = torch.ones(B, device=device)
            slot0 = torch.zeros(B, dtype=torch.long, device=device)
            dt = (1.0 - tau0) / float(steps)
            for i in range(steps):
                def _v(zz, pr):
                    tv = torch.full((2 * B,), tau0 + pr * (1.0 - tau0), device=device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v_sc, v_0c = o3_fm(
                            torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                            slot0.repeat(2), u_style.repeat(2, 1), tv,
                            drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                            layout_scale=torch.cat([l1, l1], 0),
                            d16_ref=d16_r.repeat(2, 1, 1, 1), v16_ref=v16_r.repeat(2, 1, 1, 1),
                            u_glob=u_glob.repeat(2, 1),
                        ).float().chunk(2, 0)
                    return v_0c + wj * (v_sc - v_0c)
                v1 = _v(z, i / float(steps))
                z = z + dt * _v(z + 0.5 * dt * v1, (i + 0.5) / float(steps))
            z_out = z
            _sync(device); timings["flow"] = (time.perf_counter() - tfl0) * 1000.0

        # Stage 3: S10 Decode
        _sync(device); td0 = time.perf_counter()
        _ = decode_s10(dec, z_out, batch=min(B, 8), output_cpu=False).clamp(0.0, 1.0)
        _sync(device); timings["decode"] = (time.perf_counter() - td0) * 1000.0
        timings["cached_flow_dec"] = timings["anchor"] + timings["flow"] + timings["decode"]
        timings["total_rgb2rgb"] = (timings["encode"] + timings["feature"] + timings["style"]
                                    + timings["anchor"] + timings["flow"] + timings["decode"])
        return timings

    bench = {}
    engines = ["prism1", "d6_csj_b4", "d6_full_b4", "prism_hc_r12", "prism_flow_d2", "o3_ref_fresh", "o3_ref_cached"]
    for eng in engines:
        batch_list = (1, 16) if eng != "prism_flow_d2" else (1, 8, 16)
        for bs in batch_list:
            tag = f"b{bs}"
            wu, rp = (3, 10) if bs == 1 else (2, 5)
            try:
                torch.cuda.reset_peak_memory_stats(device)
                for _ in range(wu):
                    _run_batch(eng, bs)
                torch.cuda.reset_peak_memory_stats(device)
                rows = [_run_batch(eng, bs) for _ in range(rp)]
                peak_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
                if peak_mb > 10400.0:
                    print(f"[R16 3060] {eng}_{tag} exceeded VRAM cap ({peak_mb} MB), stopping", flush=True)
                    break
                stat = _summarize(rows, bs)
                stat["peak_vram_mb"] = peak_mb
                bench[f"{eng}_{tag}"] = stat
                tot = stat["total_rgb2rgb"]
                cfd = stat["cached_flow_dec"]
                print(
                    f"[R16 3060] {eng}_{tag:3s}: E2E={tot['p50_ms_per_img']:6.2f}ms ({1000/tot['p50_ms_per_img']:4.2f} img/s) | "
                    f"Cached(Anc+Fl+Dec)={cfd['p50_ms_per_img']:6.2f}ms ({1000/cfd['p50_ms_per_img']:5.2f} img/s) | "
                    f"enc={stat['encode']['p50_ms_per_img']:5.1f} feat={stat['feature']['p50_ms_per_img']:5.1f} "
                    f"anc={stat['anchor']['p50_ms_per_img']:4.1f} flow={stat['flow']['p50_ms_per_img']:6.1f} "
                    f"dec={stat['decode']['p50_ms_per_img']:5.1f} vram={peak_mb}MB",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[R16 3060] {eng}_{tag} OOM", flush=True)
                break

    del vae, dec, dino, cnx, b4, canon, fiber_enc, p1, slf, o3_fm, o3_ad
    torch.cuda.empty_cache()
    return bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    seed_all(20261185)
    dec_res = bench_decoders(device=args.device)
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    stage_res = bench_stages_3060(bundle, bank, device=args.device)

    save_json(OUT_JSON, {
        "task": "V25-R16 3060 stage breakdown, PRISM-Flow D2 sweep & decoder benchmark",
        "updated_utc": utc(),
        "gpu": torch.cuda.get_device_name(args.device),
        "decoders": dec_res,
        "stages": stage_res,
    })
    print(f"\n=== V25 R16 完成 -> {OUT_JSON} ===", flush=True)


if __name__ == "__main__":
    main()
