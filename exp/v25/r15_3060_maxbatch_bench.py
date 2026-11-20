"""V25 R15 (3060): RTX 3060 12GB 最大安全 Batch 吞吐量基准 (img/s & ms/img).

逐引擎在 3060 12GB 物理显存内自动扫描 batch ∈ {1, 4, 8, 16, 24, 32, 48, 64},
遇 OOM 或显存逼近共享内存阈 (>10.5 GB) 即止步, 记录最快吞吐 (peak img/s) 及最稳最优 batch:
  - 全栈端到端 RGB→RGB: A0 encode + DINO/CNX 特征 + Anchor + Flow + S10 decode
  - 纯推理 (Flow+Decode, 潜/特征已缓存协议口径): Anchor(若在线) + Flow + S10 decode

涵盖全前沿引擎:
  1. prism1          (PRISM-1, 1 步)
  2. d6_csj_b4       (SLFUNet 8 步 cs_j + B4 锚)
  3. d6_full_b4      (SLFUNet 8 步 full + B4 锚)
  4. d5_csj_10s      (SLFUNet 10 步 cs_j — 代表 V22 Dual F3 / V23 HiC / DualSWD)
  5. d5_full_10s     (SLFUNet 10 步 full — 代表 V22 Peak1500 / Gate65)
  6. prism_hc_r12    (PRISM-HighContent-R12 = P1 + B4 + D6H cs_j)
  7. o3_ref_fresh    (One-Shot O3 / C2nt, 含参考特征提取)
  8. o3_ref_cached   (One-Shot O3 / C2nt, 同参考缓存)

用法 (3060):
  python -m exp.v25.r15_3060_maxbatch_bench --pool ../pseudo_gt_24h/orig_mean_fp16
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
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline, match_scene_submode
from exp.v22.core import CFGConfig, D5_SOLVER, D6_SOLVER, GateConfig, build_content_condition, effective_gate_components, make_initial_state, predict_guided_velocity, rk2_integrate
from exp.v22.ref_adapter import RefAdapter
from exp.v24.p0_m0_m2 import P1_CKPT, V20_CKPT
from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot
from exp.v25.precompute_train_latents import B4_CKPT
from exp.v25.r1_inference_ablation import D6H_SOLVER
from exp.v25.r14_e2e_benchmark import load_a0_encoder
from src.infer import canonicalizer, load_c3unet
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import load_dino_model

OUT_JSON = OUT / "r15_3060_maxbatch_bench.json"
O3_CKPT = ROOT / "runs/v24/oneshot_val/O3_rgb_cnx_gated.pt"
VRAM_CAP_MB = 10400.0  # 严控在 3060 12GB 物理显存安全区, 防 Windows 共享内存分页失真
BATCH_CANDS = (1, 4, 8, 16, 24, 32, 48, 64)


def _sync(device: str):
    torch.cuda.synchronize(device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    seed_all(20261185)
    device = args.device
    bundle = build_or_load_v19_foundation(args.pool, device=device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=device)

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

    def _run_once(eng: str, B: int) -> tuple[float, float]:
        rgb_in = torch.rand(B, 3, 512, 512, device=device)
        rgb_ref = torch.rand(1, 3, 512, 512, device=device).expand(B, -1, -1, -1)
        tid = torch.full((B,), 2, dtype=torch.long, device=device)
        eps = torch.randn(B, 4, 64, 64, device=device)

        if eng == "o3_ref_cached":
            d16_r = extract_whitened_dino_layout(dino, rgb_ref[:1], batch=1, device=device).expand(B, -1, -1, -1)
            x_r = (F.interpolate(rgb_ref[:1], size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
            v16_r = F.avg_pool2d(cnx_s2(x_r), 2, 2)
            v16_r = (v16_r / v16_r.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)).expand(B, -1, -1, -1)
            fib_r = torch.cat([extract_color_fiber_27d(rgb_ref[:1]),
                               extract_dino_rgb_moments_768d(dino, rgb_ref[:1], batch=1, device=device).to(device)], dim=-1)
            u_r_enc = fiber_enc.encode(fib_r.float()).expand(B, -1)
            u_glob = ((fib_r - o3_fib_m) / o3_fib_s).expand(B, -1)

        _sync(device); t_e2e_0 = time.perf_counter()
        z_src = _encode_chunked(rgb_in, sub=8)
        d16 = extract_whitened_dino_layout(dino, rgb_in, batch=min(B, 16), device=device)
        luma = ((z_src * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)
        y16 = coarse_y16_lcn(luma)

        if eng in ("prism1", "prism_hc_r12"):
            col27 = extract_color_fiber_27d(rgb_in)
            d768 = extract_dino_rgb_moments_768d(dino, rgb_in, batch=min(B, 16), device=device).to(device)
            fib = torch.cat([col27, d768], dim=-1)
            c64 = rgb_to_structure_64(rgb_in)
            chord = bary[tid] - fiber_enc.encode(fib.float())
        elif eng in ("o3_ref_fresh", "o3_ref_cached"):
            fib_c = torch.cat([extract_color_fiber_27d(rgb_in),
                               extract_dino_rgb_moments_768d(dino, rgb_in, batch=min(B, 16), device=device).to(device)], dim=-1)
            u_c_enc = fiber_enc.encode(fib_c.float())
            if eng == "o3_ref_fresh":
                d16_r = extract_whitened_dino_layout(dino, rgb_ref, batch=min(B, 16), device=device)
                x_r = (F.interpolate(rgb_ref, size=(256, 256), mode="bilinear", align_corners=False) - mean_im) / std_im
                v16_r = F.avg_pool2d(cnx_s2(x_r), 2, 2)
                v16_r = v16_r / v16_r.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
                fib_r = torch.cat([extract_color_fiber_27d(rgb_ref),
                                   extract_dino_rgb_moments_768d(dino, rgb_ref, batch=min(B, 16), device=device).to(device)], dim=-1)
                u_r_enc = fiber_enc.encode(fib_r.float())
                u_glob = (fib_r - o3_fib_m) / o3_fib_s
            u_style = o3_ad(u_r_enc, u_c_enc)

        if eng in ("d6_csj_b4", "d6_full_b4", "d5_csj_10s", "d5_full_10s", "prism_hc_r12"):
            _, u_style_d6 = match_scene_submode(d16, tid, bank)
            sc_src = F.normalize(F.avg_pool2d(d16.float(), 4).flatten(1), dim=-1)
            best_c = torch.einsum("bd,bkd->bk", sc_src, bank["sub_scene"].to(device).float()[tid]).argmax(dim=1)
            dsd_ref = bank["sub_dsd"].to(device).float()[tid, best_c]
            g_tau, g_layout, g_guid, _ = effective_gate_components(dsd_ref, GateConfig("original"), mid_dsd)

        # ---- Flow + S10 Decode 计时起点 (对应协议批纯推理延迟) ----
        _sync(device); t_flow_0 = time.perf_counter()
        if eng == "prism1":
            z_anc = apply_latent_chroma_shift(z_src, tid, priors, alpha=0.20)
            z_luma = torch.einsum("ij,bjhw->bihw", P_Y, z_src)
            z_out = z_anc + p1(z_anc, z_luma, c64, d16, chord)
        elif eng in ("d6_csj_b4", "d6_full_b4", "d5_csj_10s", "d5_full_10s", "prism_hc_r12"):
            is_hc = (eng == "prism_hc_r12")
            branches = "full" if "full" in eng else "cs_j"
            if eng.startswith("d5_"):
                solver = dict(D5_SOLVER)
            elif eng == "d6_full_b4":
                solver = dict(D6_SOLVER)
            else:
                solver = dict(D6H_SOLVER)
            c_can = canon(z_src, mode="c1_gentle", augment=False)
            z_anc_b4 = b4(c_can, 0.0, c_can, tid)
            if is_hc:
                z_anc_p1 = apply_latent_chroma_shift(z_src, tid, priors, alpha=0.20)
                z_luma = torch.einsum("ij,bjhw->bihw", P_Y, z_src)
                z_p1 = z_anc_p1 + p1(z_anc_p1, z_luma, c64, d16, chord)
            lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
            content = build_content_condition(y16.float(), d16.float(), g_layout)
            tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
            z = make_initial_state(eps, z_anc_b4, tau_v)
            cfg = CFGConfig(branches=branches, rescale_phi=0.48)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vel_fn = lambda zz, prog: predict_guided_velocity(
                    slf, zz, tau_v + prog * (1.0 - tau_v), content, tid, u_style_d6, lscale, g_guid, cfg, solver, prog,
                )
                z_d6 = rk2_integrate(vel_fn, z, tau_v, int(solver["steps"]))
            z_out = (0.45 * z_p1 + 0.55 * z_d6) if is_hc else z_d6
        else:
            tau0, wj, steps = 0.45, 2.2, 10
            z = (1.0 - tau0) * eps + tau0 * z_src
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

        _ = decode_s10(dec, z_out, batch=min(B, 8), output_cpu=False).clamp(0.0, 1.0)
        _sync(device)
        t_end = time.perf_counter()
        return (t_end - t_e2e_0) * 1000.0, (t_end - t_flow_0) * 1000.0

    engines = ("prism1", "d6_csj_b4", "d6_full_b4", "d5_csj_10s", "d5_full_10s",
               "prism_hc_r12", "o3_ref_fresh", "o3_ref_cached")
    results = {}
    for eng in engines:
        sweep = []
        for bs in BATCH_CANDS:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                for _ in range(2):
                    _run_once(eng, bs)
                torch.cuda.reset_peak_memory_stats(device)
                e2es, fds = [], []
                reps = 8 if bs <= 8 else 5
                for _ in range(reps):
                    e2e_ms, fd_ms = _run_once(eng, bs)
                    e2es.append(e2e_ms / bs)
                    fds.append(fd_ms / bs)
                peak_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
                if peak_mb > VRAM_CAP_MB:
                    print(f"[R15 3060] {eng:14s} bs={bs:2d} OVER_VRAM ({peak_mb}MB > {VRAM_CAP_MB}) -> stop", flush=True)
                    break
                e2e_p50 = float(np.median(e2es))
                fd_p50 = float(np.median(fds))
                row = {
                    "batch": bs, "peak_vram_mb": peak_mb,
                    "e2e_ms_per_img": round(e2e_p50, 2), "e2e_img_per_s": round(1000.0 / e2e_p50, 2),
                    "flow_dec_ms_per_img": round(fd_p50, 2), "flow_dec_img_per_s": round(1000.0 / fd_p50, 2),
                }
                sweep.append(row)
                print(f"[R15 3060] {eng:14s} bs={bs:2d} | E2E={row['e2e_ms_per_img']:6.2f}ms ({row['e2e_img_per_s']:5.2f} img/s) "
                      f"| Flow+Dec={row['flow_dec_ms_per_img']:6.2f}ms ({row['flow_dec_img_per_s']:5.2f} img/s) | vram={peak_mb}MB", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[R15 3060] {eng:14s} bs={bs:2d} OOM -> stop", flush=True)
                break
        best_e2e = max(sweep, key=lambda r: r["e2e_img_per_s"])
        best_fd = max(sweep, key=lambda r: r["flow_dec_img_per_s"])
        results[eng] = {"sweep": sweep, "best_e2e": best_e2e, "best_flow_dec": best_fd}
        save_json(OUT_JSON, {"task": "V25-R15 3060 max-batch benchmark", "updated_utc": utc(),
                             "gpu": torch.cuda.get_device_name(device), "results": results})

    print("\n=== 3060 最大安全 Batch 最快吞吐汇总 ===", flush=True)
    for eng, info in results.items():
        be, bf = info["best_e2e"], info["best_flow_dec"]
        print(f"  {eng:14s}: E2E best bs={be['batch']:2d} -> {be['e2e_img_per_s']:6.2f} img/s ({be['e2e_ms_per_img']:6.2f} ms) | "
              f"Flow+Dec best bs={bf['batch']:2d} -> {bf['flow_dec_img_per_s']:6.2f} img/s ({bf['flow_dec_ms_per_img']:6.2f} ms)", flush=True)


if __name__ == "__main__":
    main()
