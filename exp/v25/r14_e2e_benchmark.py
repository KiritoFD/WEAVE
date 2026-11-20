"""V25 R14 (4090): 最简模型正式端到端 RGB→RGB 延迟基准 + PRISMCore 双接口端到端冒烟.

落实用户收束第 4 项 (对照 A07 M1–M6/M9 必测清单):
  1. 完整 512×512 RGB → 512×512 RGB 端到端阶段拆分计时 (warmup=3, repeats=12 @ batch=1; repeats=6 @ batch=16):
     T_encode (A0 VAE) + T_feature (DINO/CNX/y16/27D) + T_style (Fiber/Adapter/Bank)
     + T_anchor (B4+canon / ChromaShift) + T_flow (PRISM-1 / cs_j 16 NFE / full 32 NFE / O3 20 NFE)
     + T_decode (S10) → 输出 P50 / P95 / mean ms/img + 峰值显存 MB.
  2. 对比引擎 (全部在同一张 4090 GPU 同口径实测):
     - prism1 (1.28M, NFE=1)
     - d6_csj_b4 (21.95M + 0.22M B4, cs_j 16 NFE ×2 = 32 img-eq)
     - d6_full_b4 (21.95M + 0.22M B4, full 16 NFE ×4 = 64 img-eq)
     - prism_hc_r12 (PRISM-1 + B4 + D6H cs_j 混合)
     - o3_ref_fresh / o3_ref_cached (22.6M One-Shot, 20 NFE ×2 = 40 img-eq, 区分首次参考 vs 同参考复用缓存)
  3. 双接口端到端冒烟 (src/model/prism_v25.py::PRISMCore):
     prism1 / d6_flow / hybrid 的 prepare().transfer(None, None, strength=0.5),
     校验 RESULT_KEYS / shape (750,4,64,64) & (750,3,512,512) / [0,1] 区间 / 重复调用确定性 / student→NotImplementedError.

用法 (4090):
  python -m exp.v25.r14_e2e_benchmark --pool ../pseudo_gt_24h/orig_mean_fp16
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

from . import EVAL_SEED, OUT, ROOT, sanitize_cuda_env, save_json, seed_all, utc
from exp.v18.structure import extract_whitened_dino_layout, rgb_to_structure_64
from exp.v19.fiber_duality import (
    StyleFiberEncoder,
    apply_latent_chroma_shift,
    build_or_load_v19_foundation,
    extract_color_fiber_27d,
    extract_dino_rgb_moments_768d,
)
from exp.v19.routed_student import SpatiallyRoutedChordNet
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline, match_scene_submode
from exp.v22.core import CFGConfig, D6_SOLVER, GateConfig, build_content_condition, effective_gate_components, make_initial_state, predict_guided_velocity, rk2_integrate
from exp.v22.ref_adapter import RefAdapter
from exp.v24.p0_m0_m2 import P1_CKPT, V20_CKPT
from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot
from exp.v25.precompute_train_latents import B4_CKPT
from exp.v25.r1_inference_ablation import D6H_SOLVER
from src.infer import canonicalizer, load_c3unet
from src.model.prism_v25 import PRISMCore, RESULT_KEYS, STUDENT_PENDING_MESSAGE
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import load_dino_model
from src.v25.config import TransferConfig

OUT_JSON = OUT / "r14_e2e_benchmark.json"
A0_ENC_CKPT = ROOT / "runs/a0/a0_encoder_only.pt"
O3_CKPT = ROOT / "runs/v24/oneshot_val/O3_rgb_cnx_gated.pt"

_SD_VAE_CFG = {
    "_class_name": "AutoencoderKL",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
    "in_channels": 3, "latent_channels": 4, "layers_per_block": 2, "norm_num_groups": 32,
    "out_channels": 3, "sample_size": 512, "scaling_factor": 0.18215,
    "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
}


def load_a0_encoder(device: str):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_config(_SD_VAE_CFG).to(device).eval()
    ck = torch.load(A0_ENC_CKPT, map_location="cpu", weights_only=False)
    vae.load_state_dict(ck["state_dict"], strict=False)
    vae = vae.half()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _summarize(samples: list[dict[str, float]], batch: int) -> dict:
    keys = list(samples[0].keys())
    out = {"batch": batch}
    for k in keys:
        vals = np.array([s[k] for s in samples], dtype=np.float64) / float(batch)
        out[k] = {
            "mean_ms_per_img": round(float(vals.mean()), 2),
            "p50_ms_per_img": round(float(np.percentile(vals, 50)), 2),
            "p95_ms_per_img": round(float(np.percentile(vals, 95)), 2),
        }
    return out


def _sync(device: str):
    torch.cuda.synchronize(device)


@torch.no_grad()
def bench_engines(bundle, bank, device: str = "cuda", warmup: int = 3, repeats: int = 12) -> dict:
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

    o3_fm, o3_ad, o3_fib_m, o3_fib_s = None, None, None, None
    if O3_CKPT.exists():
        ck_o3 = torch.load(O3_CKPT, map_location="cpu", weights_only=False)
        o3_fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device).eval()
        o3_fm.load_state_dict(ck_o3["fm"], strict=True)
        o3_ad = RefAdapter(dim=128).to(device).eval()
        o3_ad.load_state_dict(ck_o3["adapter"], strict=True)
        o3_fib_m, o3_fib_s = ck_o3["fib_mean"].to(device), ck_o3["fib_std"].to(device)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    mid_dsd = float(bank["mid_dsd"])

    def _run_batch(engine_name: str, B: int) -> dict:
        rgb_in = torch.rand(B, 3, 512, 512, device=device)
        rgb_ref = torch.rand(B, 3, 512, 512, device=device)
        tid = torch.full((B,), 2, dtype=torch.long, device=device)
        eps = torch.randn(B, 4, 64, 64, device=device)
        timings = {}

        # Stage 1: Encode
        _sync(device); t0 = time.perf_counter()
        z_src = (vae.encode((rgb_in.half() * 2.0 - 1.0)).latent_dist.mean * 0.18215).float()
        _sync(device); timings["encode"] = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Feature + Style + Anchor + Flow
        if engine_name == "prism1":
            _sync(device); tf0 = time.perf_counter()
            d16 = extract_whitened_dino_layout(dino, rgb_in, batch=B, device=device)
            col27 = extract_color_fiber_27d(rgb_in)
            d768 = extract_dino_rgb_moments_768d(dino, rgb_in, batch=B, device=device).to(device)
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
            d16 = extract_whitened_dino_layout(dino, rgb_in, batch=B, device=device)
            luma = ((z_src * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)
            y16 = coarse_y16_lcn(luma)
            if is_hc:
                col27 = extract_color_fiber_27d(rgb_in)
                d768 = extract_dino_rgb_moments_768d(dino, rgb_in, batch=B, device=device).to(device)
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
            vel_fn = lambda zz, prog: predict_guided_velocity(
                slf, zz, tau_v + prog * (1.0 - tau_v), content, tid, u_style, lscale, g_guid, cfg, solver, prog,
            )
            z_d6 = rk2_integrate(vel_fn, z, tau_v, int(solver["steps"]))
            z_out = (0.45 * z_p1 + 0.55 * z_d6) if is_hc else z_d6
            _sync(device); timings["flow"] = (time.perf_counter() - tfl0) * 1000.0

        elif engine_name in ("o3_ref_fresh", "o3_ref_cached"):
            assert o3_fm is not None
            cached_ref = (engine_name == "o3_ref_cached")
            if cached_ref:
                d16_r = extract_whitened_dino_layout(dino, rgb_ref, batch=B, device=device)
                x_r = (F.interpolate(rgb_ref, size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
                v16_r = F.avg_pool2d(cnx_s2(x_r), 2, 2)
                v16_r = v16_r / v16_r.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
                fib_r = torch.cat([extract_color_fiber_27d(rgb_ref),
                                   extract_dino_rgb_moments_768d(dino, rgb_ref, batch=B, device=device).to(device)], dim=-1)
                u_r_enc = fiber_enc.encode(fib_r.float())
                u_glob = (fib_r - o3_fib_m) / o3_fib_s

            _sync(device); tf0 = time.perf_counter()
            y16 = coarse_y16_lcn(((z_src * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0))
            d16 = extract_whitened_dino_layout(dino, rgb_in, batch=B, device=device)
            fib_c = torch.cat([extract_color_fiber_27d(rgb_in),
                               extract_dino_rgb_moments_768d(dino, rgb_in, batch=B, device=device).to(device)], dim=-1)
            if not cached_ref:
                d16_r = extract_whitened_dino_layout(dino, rgb_ref, batch=B, device=device)
                x_r = (F.interpolate(rgb_ref, size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
                v16_r = F.avg_pool2d(cnx_s2(x_r), 2, 2)
                v16_r = v16_r / v16_r.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
                fib_r = torch.cat([extract_color_fiber_27d(rgb_ref),
                                   extract_dino_rgb_moments_768d(dino, rgb_ref, batch=B, device=device).to(device)], dim=-1)
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
        else:
            raise ValueError(engine_name)

        # Stage 3: S10 Decode
        _sync(device); td0 = time.perf_counter()
        _ = decode_s10(dec, z_out, batch=min(B, 16), output_cpu=False).clamp(0.0, 1.0)
        _sync(device); timings["decode"] = (time.perf_counter() - td0) * 1000.0
        timings["total_rgb2rgb"] = (timings["encode"] + timings["feature"] + timings["style"]
                                    + timings["anchor"] + timings["flow"] + timings["decode"])
        return timings

    engines = ["prism1", "d6_csj_b4", "d6_full_b4", "prism_hc_r12"]
    if o3_fm is not None:
        engines.extend(["o3_ref_fresh", "o3_ref_cached"])

    bench = {}
    for eng in engines:
        for bs, wu, rp, tag in ((1, warmup, repeats, "b1"), (16, 2, 6, "b16")):
            if eng.startswith("o3_ref") and bs == 16:
                continue  # O3 部署口径主要为单图 batch=1 (避免 32-batch CFG 显存偏离部署协议)
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(wu):
                _run_batch(eng, bs)
            torch.cuda.reset_peak_memory_stats(device)
            rows = [_run_batch(eng, bs) for _ in range(rp)]
            peak_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
            stat = _summarize(rows, bs)
            stat["peak_vram_mb"] = peak_mb
            bench[f"{eng}_{tag}"] = stat
            tot = stat["total_rgb2rgb"]
            print(f"[R14 BENCH] {eng}_{tag:3s}: p50={tot['p50_ms_per_img']:7.2f}ms/img "
                  f"p95={tot['p95_ms_per_img']:7.2f} flow={stat['flow']['p50_ms_per_img']:6.2f} "
                  f"enc={stat['encode']['p50_ms_per_img']:5.2f} feat={stat['feature']['p50_ms_per_img']:5.2f} "
                  f"dec={stat['decode']['p50_ms_per_img']:5.2f} vram={peak_mb}MB", flush=True)

    del vae, dec, dino, cnx, b4, canon, fiber_enc, p1, slf, o3_fm, o3_ad
    torch.cuda.empty_cache()
    return bench


def run_dual_interface_smoke(pool: str, device: str = "cuda") -> dict:
    """PRISMCore 双接口端到端 GPU 冒烟 (prism1 / d6_flow / hybrid / student gate / determinism)."""
    smoke = {}
    # 1. student pending gate
    stu = PRISMCore(TransferConfig(engine="student"))
    try:
        stu.transfer(None, None, strength=0.5)
        raise AssertionError("student engine must raise NotImplementedError")
    except NotImplementedError as exc:
        assert STUDENT_PENDING_MESSAGE in str(exc)
        smoke["student_gate"] = "PASS (NotImplementedError raised)"

    # 2. PRISMCore available engines (prepare once, transfer with shared protocol resources)
    shared_core = PRISMCore(TransferConfig(engine="prism1")).prepare(pool=pool, device=device)
    resources = shared_core._resources
    for eng in ("prism1", "d6_flow", "hybrid"):
        core = PRISMCore(TransferConfig(engine=eng), runtime=shared_core._runtime, resources=resources)
        t0 = time.perf_counter()
        out1 = core.transfer(None, None, strength=0.5)
        dt = time.perf_counter() - t0
        for k in RESULT_KEYS:
            assert k in out1, f"missing key {k}"
        z, rgb = out1["z"], out1["rgb01"]
        assert tuple(z.shape) == (750, 4, 64, 64), f"bad z shape {tuple(z.shape)}"
        assert tuple(rgb.shape) == (750, 3, 512, 512), f"bad rgb shape {tuple(rgb.shape)}"
        assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(rgb).all())
        assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0

        # Determinism check
        out2 = core.transfer(None, None, strength=0.5)
        det = bool(torch.equal(out1["z"], out2["z"])) and bool(torch.equal(out1["rgb01"], out2["rgb01"]))
        assert det, f"{eng} transfer not deterministic"
        smoke[f"PRISMCore_{eng}"] = {
            "status": "PASS",
            "knob": out1["knob"],
            "knob_value": round(float(out1["knob_value"]), 5),
            "config_hash": core.config_hash[:12],
            "deterministic": det,
            "wall_s_750": round(dt, 2),
        }
        print(f"[R14 SMOKE] PRISMCore({eng}): PASS knob={out1['knob']}={out1['knob_value']:.4f} det={det} ({dt:.1f}s)", flush=True)
    return smoke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    seed_all(EVAL_SEED)
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)

    bench = bench_engines(bundle, bank, device=args.device)
    smoke = run_dual_interface_smoke(args.pool, device=args.device)

    save_json(OUT_JSON, {
        "task": "V25-R14 formal E2E RGB->RGB benchmark & PRISMCore smoke",
        "updated_utc": utc(),
        "gpu": torch.cuda.get_device_name(args.device),
        "benchmark": bench,
        "smoke": smoke,
    })
    print(f"\n=== V25 R14 完成 -> {OUT_JSON} ===", flush=True)


if __name__ == "__main__":
    main()
