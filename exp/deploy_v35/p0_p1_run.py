"""P0 (TRT FP16 + INT8 probes) & P1 (TAESD zero-train + FLOPs-aware A0 pruning design) on RTX 3060."""
from __future__ import annotations

import json
import copy
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from exp.deploy_v35 import BUNDLE_DEFAULT_DIR, OPT_OUT_DIR
from exp.deploy_v35.a0_prune import A0_ENC_WIDTHS, StudentEncoder, prune_encoder, taylor_encoder_importance
from exp.deploy_v35.bundle import load_a0_encoder_from_file, load_s10b_from_file
from exp.deploy_v35.eval_gate import paired_cluster_bootstrap
from exp.deploy_v35.parity import load_rgb01_from_rel, make_deterministic_eps, select_stratified_tasks
from exp.deploy_v35.pipeline import CleaveDeployPipeline, make_optimized_pipeline


def _load_16_suite() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tasks16 = select_stratified_tasks(16)
    rgb16 = torch.stack([load_rgb01_from_rel(t["source_rel_path"]) for t in tasks16], 0).cuda()
    tids = torch.tensor([int(t["target_style_id"]) for t in tasks16], device="cuda", dtype=torch.long)
    eps16 = torch.stack([make_deterministic_eps(t["task_id"]) for t in tasks16], 0).cuda()
    return rgb16, tids, eps16
from exp.deploy_v35.torch_opt import FastA0Encoder, FastS10BDecoder, enable_persistent_inductor_cache
from exp.deploy_v35.trt_runtime import TrtRunner, bench_gpu_callable, build_trt_engine, export_onnx_clean
from src.utils.runtime import REPO
from exp.v20.slf_flow import project_latent_y16_lcn


WIKIART_TRAIN_DIR = Path(os.environ.get("WEAVE_DISTILL_TRAIN_ROOT", "datasets/wikiart_new_512"))


def _load_train_images(n: int = 128, seed: int = 0) -> torch.Tensor:
    from exp.deploy_v35.distill.data import build_manifest
    manifest = build_manifest(WIKIART_TRAIN_DIR, OPT_OUT_DIR / 'int8_train_audit', n, 4, seed)
    files = [WIKIART_TRAIN_DIR / r['rel'] for r in manifest['splits']['train']]
    xs = []
    for file in files:
        im = Image.open(file).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        xs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(xs, 0)


@torch.inference_mode()
def run_p0_trt_and_int8(report_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    trt_dir = report_dir / "trt_engines"
    trt_dir.mkdir(parents=True, exist_ok=True)
    int8_dir = trt_dir / 'audited_int8'
    int8_dir.mkdir(exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # 1. FP32 base modules for clean ONNX export (zero-loss weights)
    a0_raw = load_a0_encoder_from_file(BUNDLE_DEFAULT_DIR / "a0_encoder.pt", device="cuda", dtype=torch.float32)
    a0_f32 = FastA0Encoder(a0_raw, 0.18215).float().cuda().eval()
    s10_raw = load_s10b_from_file(BUNDLE_DEFAULT_DIR / "s10b_decoder.pt", device="cuda", dtype=torch.float32)
    s10_f32 = FastS10BDecoder(s10_raw, 0.18215, dtype=torch.float32, channels_last=False).cuda().eval()

    # Use optimized PyTorch pipeline (without compile first) for clean E-Core & HoistedPCore export
    pipe_eager = make_optimized_pipeline(BUNDLE_DEFAULT_DIR, load_pcore=True, compile_mode="")
    # Cast fast_ecore & fast_pcore modules to float32 copy ONLY for ONNX export, keep cuda eager for reference
    pipe_ref_opt = make_optimized_pipeline(BUNDLE_DEFAULT_DIR, load_pcore=True, compile_mode="")

    # Calibration tensors ONLY from WikiArt train set
    calib_imgs = _load_train_images(64, seed=42)
    with torch.no_grad():
        calib_zs = torch.cat([a0_f32(calib_imgs[i : i + 8].cuda()).cpu() for i in range(0, 64, 8)], 0)

    # Downstream 16 stratified test tasks (reference pipeline output)
    rgb16, tids, eps16 = _load_16_suite()

    with torch.no_grad():
        ref_e_imgs = []
        ref_p_imgs = []
        ref_zs_list = []
        ref_edit_latents = []
        for i in range(16):
            r_i = rgb16[i : i + 1]
            t_i = tids[i : i + 1]
            e_i = eps16[i : i + 1]
            src_i = pipe_ref_opt.prepare_source(r_i, need_y16=True)
            ref_zs_list.append(src_i["z_s"].float())
            re_i, de_i = pipe_ref_opt.run_e_only(r_i, t_i, 0.60, e_i, cached_src=src_i, return_intermediates=True)
            rp_i, dp_i = pipe_ref_opt.run_e_to_p(r_i, t_i, 1.00, e_i, cached_src=src_i, return_intermediates=True)
            ref_edit_latents.extend([de_i['z_out'], dp_i['zo']])
            ref_e_imgs.append(re_i)
            ref_p_imgs.append(rp_i)
        ref_zs16 = torch.cat(ref_zs_list, 0)
        ref_e16 = torch.cat(ref_e_imgs, 0)
        ref_p16 = torch.cat(ref_p_imgs, 0)
        edited_zs = torch.cat(ref_edit_latents, 0)

    # Module TRT FP16 & INT8 probes
    mod_bench: dict[str, Any] = {}
    for bs in (1, 4):
        print(f'[P0 modules] B={bs}', flush=True)
        rgb = torch.rand(bs, 3, 512, 512, device="cuda", dtype=torch.float32)
        z = torch.randn(bs, 4, 64, 64, device="cuda", dtype=torch.float32)

        a0_onnx = export_onnx_clean(a0_f32, (rgb,), trt_dir / f"a0_b{bs}.onnx", ["rgb"], ["z_s"])
        s10_onnx = export_onnx_clean(s10_f32, (z,), trt_dir / f"s10_b{bs}.onnx", ["zo"], ["rgb_out"])

        # A0 & S10B FP16
        _, bt_a0_fp16 = build_trt_engine(a0_onnx, trt_dir / f"a0_b{bs}_fp16.plan", precision="fp16")
        _, bt_s10_fp16 = build_trt_engine(s10_onnx, trt_dir / f"s10_b{bs}_fp16.plan", precision="fp16")
        run_a0_fp16 = TrtRunner(trt_dir / f"a0_b{bs}_fp16.plan", ["rgb"], ["z_s"])
        run_s10_fp16 = TrtRunner(trt_dir / f"s10_b{bs}_fp16.plan", ["zo"], ["rgb_out"])

        ms_a0_fp16 = bench_gpu_callable(lambda: run_a0_fp16(rgb), bs)
        ms_s10_fp16 = bench_gpu_callable(lambda: run_s10_fp16(z), bs)

        row: dict[str, Any] = {
            "a0_fp16_ms": round(ms_a0_fp16, 3),
            "s10_fp16_ms": round(ms_s10_fp16, 3),
        }

        # INT8 probes at B=4 only (most representative throughput batch)
        if bs == 4:
            for pmode in ("int8_full", "int8_mixed"):
                try:
                    _, _ = build_trt_engine(
                        a0_onnx,
                        int8_dir / f"a0_b4_{pmode}.plan",
                        precision=pmode,
                        calib_samples=calib_imgs,
                        calib_batch=4,
                        calib_cache=int8_dir / f"a0_b4_{pmode}.calib",
                    )
                    r_a0_i8 = TrtRunner(int8_dir / f"a0_b4_{pmode}.plan", ["rgb"], ["z_s"])
                    ms_i8 = bench_gpu_callable(lambda: r_a0_i8(rgb), 4)
                    # Measure z_s drift + downstream 16-sample E & P mae255!
                    cand_zs = torch.cat([r_a0_i8(rgb16[i : i + 4]) for i in range(0, 16, 4)], 0)
                    cos_z = F.cosine_similarity(cand_zs.flatten(), ref_zs16.flatten(), dim=0).item()
                    mae_z = (cand_zs - ref_zs16).abs().mean().item()
                    ee_list, ep_list = [], []
                    for i in range(16):
                        src_mod = pipe_ref_opt.prepare_source(rgb16[i : i + 1], need_y16=True)
                        src_mod["z_s"] = cand_zs[i : i + 1]
                        src_mod["y16"] = project_latent_y16_lcn(src_mod["z_s"], pipe_ref_opt.w_luma, pipe_ref_opt.b_luma)
                        ee, _ = pipe_ref_opt.run_e_only(rgb16[i : i + 1], tids[i : i + 1], 0.60, eps16[i : i + 1], cached_src=src_mod, return_intermediates=True)
                        ep, _ = pipe_ref_opt.run_e_to_p(rgb16[i : i + 1], tids[i : i + 1], 1.00, eps16[i : i + 1], cached_src=src_mod, return_intermediates=True)
                        ee_list.append(ee)
                        ep_list.append(ep)
                    e_mae255 = (torch.cat(ee_list, 0) - ref_e16).abs().mean().item() * 255.0
                    p_mae255 = (torch.cat(ep_list, 0) - ref_p16).abs().mean().item() * 255.0
                    row[f"a0_{pmode}"] = {
                        "ms_img": round(ms_i8, 3),
                        "z_cos": round(cos_z, 5),
                        "z_mae": round(mae_z, 5),
                        "e_mae255": round(e_mae255, 2),
                        "p_mae255": round(p_mae255, 2),
                    }
                except Exception as exc:
                    row[f"a0_{pmode}"] = {"error": str(exc)}

                try:
                    _, _ = build_trt_engine(
                        s10_onnx,
                        int8_dir / f"s10_b4_{pmode}.plan",
                        precision=pmode,
                        calib_samples=calib_zs,
                        calib_batch=4,
                        calib_cache=int8_dir / f"s10_b4_{pmode}.calib",
                    )
                    r_s10_i8 = TrtRunner(int8_dir / f"s10_b4_{pmode}.plan", ["zo"], ["rgb_out"])
                    ms_s_i8 = bench_gpu_callable(lambda: r_s10_i8(z), 4)
                    with torch.no_grad():
                        s_ref_out = torch.cat([s10_f32(x) for x in edited_zs.split(4)], 0)
                        s_i8_out = torch.cat([r_s10_i8(x) for x in edited_zs.split(4)], 0)
                    s_mae255 = (s_i8_out - s_ref_out).abs().mean().item() * 255.0
                    s_cos = F.cosine_similarity(s_i8_out.flatten(), s_ref_out.flatten(), dim=0).item()
                    row[f"s10_{pmode}"] = {
                        "ms_img": round(ms_s_i8, 3),
                        "mae255": round(s_mae255, 2),
                        "cos": round(s_cos, 5),
                    }
                except Exception as exc:
                    row[f"s10_{pmode}"] = {"error": str(exc)}

        mod_bench[f"B{bs}"] = row
        (report_dir / "p0_partial.json").write_text(json.dumps(mod_bench, indent=2), encoding="utf-8")

    # Export P-Core & E-Core TRT FP16 (FP32 wrapper export -> TRT FP16)
    pcore_f32 = pipe_eager.fast_pcore.float().cuda().eval()
    pcore_f32.dtype = torch.float32
    ecore_f32 = pipe_eager.fast_ecore.float().cuda().eval()
    ecore_f32.dtype = torch.float32
    pcore_trt: dict[int, Any] = {}
    ecore_trt: dict[int, Any] = {}
    for bs in (1, 4):
        # Use the deployed conditioning contract (c64 has four channels).
        source = pipe_ref_opt.prepare_source(rgb16[:bs], need_y16=True)
        z_s, c64, q16, y16 = [source[k].float() for k in ('z_s', 'c64', 'q16', 'y16')]
        tid = tids[:bs]
        lam = torch.full((bs,), 0.60, device="cuda", dtype=torch.float32)
        eps = eps16[:bs]

        # E-Core
        try:
            ec_onnx = export_onnx_clean(ecore_f32, (z_s, c64, q16, tid, lam, eps), trt_dir / f"ecore_b{bs}.onnx", ["z_s", "c64", "q16", "tid", "lam", "eps"], ["ze"])
            _, _ = build_trt_engine(ec_onnx, trt_dir / f"ecore_b{bs}_fp16.plan", precision="fp16")
            r_ec = TrtRunner(trt_dir / f"ecore_b{bs}_fp16.plan", ["z_s", "c64", "q16", "tid", "lam", "eps"], ["ze"])
            ms_ec = bench_gpu_callable(lambda: r_ec(z_s, c64, q16, tid, lam, eps), bs)
            ecore_trt[bs] = {"runner": r_ec, "ms_img": round(ms_ec, 3)}
        except Exception as exc:
            ecore_trt[bs] = {"error": str(exc)}

        # HoistedPCoreStep single 4B call (trajectory calls 7x)
        try:
            ds4 = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.bool).repeat_interleave(bs)
            dc4 = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.bool).repeat_interleave(bs)
            lsc4 = torch.tensor([1.0, 0.0, 1.0, 0.0], device="cuda").repeat_interleave(bs)
            u_st = torch.randn(bs, 417, device="cuda", dtype=torch.float32)
            gc1, bc1, gc2, bc2, cb4 = pcore_f32.prepare_traj(y16, q16, tid, u_st, ds4, dc4, lsc4, 1.0)
            t1b = torch.full((bs,), 0.5, device="cuda", dtype=torch.float32)
            pc_onnx = export_onnx_clean(
                pcore_f32,
                (z_s, t1b, cb4.float(), gc1.float(), bc1.float(), gc2.float(), bc2.float()),
                trt_dir / f"pcore_b{bs}.onnx",
                ["z", "t_1b", "cb4", "gc1", "bc1", "gc2", "bc2"],
                ["v4b"],
            )
            _, _ = build_trt_engine(pc_onnx, trt_dir / f"pcore_b{bs}_fp16.plan", precision="fp16")
            r_pc = TrtRunner(trt_dir / f"pcore_b{bs}_fp16.plan", ["z", "t_1b", "cb4", "gc1", "bc1", "gc2", "bc2"], ["v4b"])
            cb4f, gc1f, bc1f, gc2f, bc2f = cb4.float(), gc1.float(), bc1.float(), gc2.float(), bc2.float()
            ms_pc_step = bench_gpu_callable(lambda: r_pc(z_s, t1b, cb4f, gc1f, bc1f, gc2f, bc2f), bs)
            pcore_trt[bs] = {"runner": r_pc, "step_ms_img": round(ms_pc_step, 3), "traj_7x_est_ms": round(ms_pc_step * 7.0, 2)}
        except Exception as exc:
            pcore_trt[bs] = {"error": str(exc)}

    out["modules"] = mod_bench
    out["ecore_trt"] = {k: {kk: vv for kk, vv in v.items() if kk != "runner"} for k, v in ecore_trt.items()}
    out["pcore_trt"] = {k: {kk: vv for kk, vv in v.items() if kk != "runner"} for k, v in pcore_trt.items()}

    # End-to-End Hybrid TRT FP16 (TRT A0 + compiled PyTorch Cond + TRT ECore/PCore + TRT S10B) parity & speed
    pipe_hybrid = make_optimized_pipeline(BUNDLE_DEFAULT_DIR, load_pcore=True, compile_mode="reduce-overhead", compile_batches=(1, 4))
    e2e_trt: dict[str, Any] = {}
    for bs in (1, 4):
        run_a0 = TrtRunner(trt_dir / f"a0_b{bs}_fp16.plan", ["rgb"], ["z_s"])
        run_s10 = TrtRunner(trt_dir / f"s10_b{bs}_fp16.plan", ["zo"], ["rgb_out"])
        run_ec = ecore_trt.get(bs, {}).get("runner")
        run_pc = pcore_trt.get(bs, {}).get("runner")

        # Install lightweight overrides just for this benchmark loop
        rgb_x = rgb16[:bs]
        tid_x = tids[:bs]
        eps_x = eps16[:bs]

        def _e_hybrid():
            z_s = run_a0(rgb_x)
            src = pipe_hybrid._extract_conditioning(rgb_x, need_y16=False, z_s=z_s)
            if run_ec is not None:
                ze = run_ec(src["z_s"], src["c64"], src["q16"], tid_x, torch.full((bs,), 0.60, device="cuda"), eps_x)
            else:
                ze = pipe_hybrid._run_ecore(src["z_s"], src["c64"], src["q16"], tid_x, 0.60, eps_x)
            return run_s10(ze)

        # Warm cond subgraph
        for _ in range(5):
            _ = _e_hybrid()
        ms_e = bench_gpu_callable(_e_hybrid, bs, warmup=8, iters=40)

        # Hot cached E-only
        z_s0 = run_a0(rgb_x)
        src0 = pipe_hybrid._extract_conditioning(rgb_x, need_y16=True, z_s=z_s0)

        def _hot_e():
            ze = run_ec(src0["z_s"], src0["c64"], src0["q16"], tid_x, torch.full((bs,), 0.60, device="cuda"), eps_x) if run_ec else pipe_hybrid._run_ecore(src0["z_s"], src0["c64"], src0["q16"], tid_x, 0.60, eps_x)
            return run_s10(ze)

        ms_hot_e = bench_gpu_callable(_hot_e, bs, warmup=8, iters=40)

        # Hybrid E->P (swap a0 & s10, and pcore step if available)
        orig_a0, orig_s10 = pipe_hybrid._encode_a0, pipe_hybrid._decode_s10
        orig_ecore = pipe_hybrid._run_ecore
        pipe_hybrid._encode_a0 = lambda x: run_a0(x)
        pipe_hybrid._decode_s10 = lambda z: run_s10(z)
        if run_ec is not None:
            pipe_hybrid._run_ecore = lambda z,c,q,t,l,e: run_ec(z,c,q,t,torch.full((len(z),),l,device=z.device),e)
        if run_pc is not None:
            orig_get = pipe_hybrid._get_callable

            def _hyb_get(kind, b_cur, fn_obj):
                if kind == "pcore" and b_cur == bs:
                    return lambda zz, t1b, cb4, gc1, bc1, gc2, bc2: run_pc(zz.float(), t1b.float(), cb4.float(), gc1.float(), bc1.float(), gc2.float(), bc2.float())
                return orig_get(kind, b_cur, fn_obj)

            pipe_hybrid._get_callable = _hyb_get
        for _ in range(3):
            _ = pipe_hybrid.run_e_to_p(rgb_x, tid_x, 1.00, eps_x)
        ms_p = bench_gpu_callable(lambda: pipe_hybrid.run_e_to_p(rgb_x, tid_x, 1.00, eps_x), bs, warmup=6, iters=25)
        ms_hot_p = bench_gpu_callable(lambda: pipe_hybrid.run_e_to_p(rgb_x, tid_x, 1.00, eps_x, cached_src=src0), bs, warmup=6, iters=25)

        # Check 16-sample parity at bs=4
        e2e_trt[f"B{bs}"] = {
            "e_only_ms": round(ms_e, 2),
            "e_only_ips": round(1000.0 / ms_e, 2),
            "e_hot_ms": round(ms_hot_e, 2),
            "e_hot_ips": round(1000.0 / ms_hot_e, 2),
            "e_to_p_ms": round(ms_p, 2),
            "e_to_p_ips": round(1000.0 / ms_p, 2),
            "p_hot_ms": round(ms_hot_p, 2),
            "p_hot_ips": round(1000.0 / ms_hot_p, 2),
        }
        if bs == 4:
            outs_e = torch.cat([pipe_hybrid.run_e_only(rgb16[i : i + 4], tids[i : i + 4], 0.60, eps16[i : i + 4]) for i in range(0, 16, 4)], 0)
            outs_p = torch.cat([pipe_hybrid.run_e_to_p(rgb16[i : i + 4], tids[i : i + 4], 1.00, eps16[i : i + 4]) for i in range(0, 16, 4)], 0)
            e2e_trt["B4_parity"] = {
                "e_mae255": round((outs_e - ref_e16).abs().mean().item() * 255.0, 3),
                "p_mae255": round((outs_p - ref_p16).abs().mean().item() * 255.0, 3),
            }
        pipe_hybrid._encode_a0, pipe_hybrid._decode_s10 = orig_a0, orig_s10
        pipe_hybrid._run_ecore = orig_ecore
        if run_pc is not None:
            pipe_hybrid._get_callable = orig_get

    out["e2e_trt_hybrid"] = e2e_trt
    return out


@torch.inference_mode()
def _run_taesd_probe(report_dir: Path) -> dict[str, Any]:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    out: dict[str, Any] = {}

    # 1. TAESD Zero-Train Probe
    from diffusers import AutoencoderTiny
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd").cuda().eval()
    taesd_enc = taesd.encoder.float().cuda().eval()
    t_params = sum(p.numel() for p in taesd_enc.parameters())
    rgb1 = torch.rand(1, 3, 512, 512, device="cuda")
    with FlopCounterMode(display=False) as fm:
        _ = taesd.encode(rgb1).latents
    t_flops_gf = fm.get_total_flops() / 1e9
    ms_taesd_b1 = bench_gpu_callable(lambda: taesd.encode(rgb1).latents, 1)
    rgb4 = torch.rand(4, 3, 512, 512, device="cuda")
    ms_taesd_b4 = bench_gpu_callable(lambda: taesd.encode(rgb4).latents, 4)

    # Check both conventions on 16 test samples (TAESD expects [0,1] natively or [-1,1] in diffusers AutoencoderTiny)
    pipe = make_optimized_pipeline(BUNDLE_DEFAULT_DIR, load_pcore=True, compile_mode="")
    rgb16, tids, eps16 = _load_16_suite()

    with torch.no_grad():
        ref_zs, ref_e, ref_p = [], [], []
        for i in range(16):
            s = pipe.prepare_source(rgb16[i : i + 1], need_y16=True)
            ref_zs.append(s["z_s"])
            re_i, _ = pipe.run_e_only(rgb16[i : i + 1], tids[i : i + 1], 0.60, eps16[i : i + 1], cached_src=s, return_intermediates=True)
            rp_i, _ = pipe.run_e_to_p(rgb16[i : i + 1], tids[i : i + 1], 1.00, eps16[i : i + 1], cached_src=s, return_intermediates=True)
            ref_e.append(re_i)
            ref_p.append(rp_i)
        z_ref = torch.cat(ref_zs, 0)
        e_ref = torch.cat(ref_e, 0)
        p_ref = torch.cat(ref_p, 0)

        # Report external scaling alternatives without choosing a convention on test results.
        z_t_m11_raw = torch.cat([taesd.encode(x * 2.0 - 1.0).latents for x in rgb16.split(2)])
        z_t_m11_scaled = z_t_m11_raw * 0.18215  # A0 storage convention, independent of Tiny config scaling=1
        z_t_01_raw = torch.cat([taesd_enc(x) for x in rgb16.split(2)])

    taesd_variants = {}
    for vname, zcand in [("m11_scaled", z_t_m11_scaled), ("m11_raw", z_t_m11_raw), ("wrong_rgb01_input_control", z_t_01_raw)]:
        cos_z = F.cosine_similarity(zcand.flatten(), z_ref.flatten(), dim=0).item()
        mae_z = (zcand - z_ref).abs().mean().item()
        ee, ep = [], []
        with torch.no_grad():
            for i in range(16):
                s = pipe.prepare_source(rgb16[i : i + 1], need_y16=True)
                s["z_s"] = zcand[i : i + 1]
                s["y16"] = project_latent_y16_lcn(s["z_s"], pipe.w_luma, pipe.b_luma)
                re_i, _ = pipe.run_e_only(rgb16[i : i + 1], tids[i : i + 1], 0.60, eps16[i : i + 1], cached_src=s, return_intermediates=True)
                rp_i, _ = pipe.run_e_to_p(rgb16[i : i + 1], tids[i : i + 1], 1.00, eps16[i : i + 1], cached_src=s, return_intermediates=True)
                ee.append(re_i)
                ep.append(rp_i)
        e_mae = (torch.cat(ee, 0) - e_ref).abs().mean().item() * 255.0
        p_mae = (torch.cat(ep, 0) - p_ref).abs().mean().item() * 255.0
        taesd_variants[vname] = {
            "z_mean": round(zcand.mean().item(), 4),
            "z_std": round(zcand.std().item(), 4),
            "z_cos": round(cos_z, 5),
            "z_mae": round(mae_z, 5),
            "e_mae255": round(e_mae, 2),
            "p_mae255": round(p_mae, 2),
        }

    out["taesd_probe"] = {
        "config_scaling_factor": float(taesd.config.scaling_factor),
        "selection": "diagnostic only; no TAESD variant promoted from test probes",
        "params_M": round(t_params / 1e6, 3),
        "flops_GF": round(t_flops_gf, 2),
        "torch_eager_b1_ms": round(ms_taesd_b1, 2),
        "torch_eager_b4_ms": round(ms_taesd_b4, 2),
        "ref_z_mean": round(z_ref.mean().item(), 4),
        "ref_z_std": round(z_ref.std().item(), 4),
        "variants": taesd_variants,
    }

    (report_dir / "p1_taesd_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


@torch.inference_mode()
def run_p1_taesd_and_prune_design(report_dir: Path) -> dict[str, Any]:
    try:
        out = _run_taesd_probe(report_dir)
    except OSError as exc:
        # An unavailable optional pretrained model does not invalidate A0 pruning.
        out = {'taesd_probe': {'status': 'unavailable', 'reason': str(exc)}}
        (report_dir / "p1_taesd_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    torch.cuda.empty_cache()
    rgb1 = torch.rand(1, 3, 512, 512, device='cuda')
    rgb4 = torch.rand(4, 3, 512, 512, device='cuda')

    # 2. A0 Encoder Pruning Design + Exact Identity Verification + FLOPs/Speed Profiling
    a0_raw = load_a0_encoder_from_file(BUNDLE_DEFAULT_DIR / "a0_encoder.pt", device="cuda", dtype=torch.float32)
    a0_ref = FastA0Encoder(a0_raw, 0.18215).cuda().eval()
    stu_id, _ = prune_encoder(a0_raw, A0_ENC_WIDTHS, importance=None)
    stu_id = stu_id.cuda().eval()
    with torch.no_grad():
        id_err = (a0_ref(rgb4[:2]) - stu_id(rgb4[:2])).abs().max().item()
    assert id_err < 1e-4, f"Identity encoder pruning mismatch: {id_err}"

    trt_dir = report_dir / "trt_engines"
    enable_persistent_inductor_cache(OPT_OUT_DIR / "inductor_cache")
    prune_candidates = [
        ("A0_prune_20M", [96, 192, 384, 384]),
        ("A0_prune_FLOPs10M", [64, 128, 256, 320]),
    ]
    prune_table = []
    for name, widths in prune_candidates:
        stu, meta = prune_encoder(a0_raw, widths, importance=None)
        stu_f32 = stu.cuda().eval()
        with FlopCounterMode(display=False) as fm:
            _ = stu_f32(rgb1)
        gf = fm.get_total_flops() / 1e9

        # Torch compile reduce-overhead fp16 benchmark
        stu_f16 = StudentEncoder(copy.deepcopy(stu.encoder).half(), copy.deepcopy(stu.qc_mean).half(), 0.18215).cuda().eval()
        stu_c1 = torch.compile(stu_f16, mode="reduce-overhead", dynamic=False)
        stu_c4 = torch.compile(stu_f16, mode="reduce-overhead", dynamic=False)

        def _c1():
            torch.compiler.cudagraph_mark_step_begin()
            return stu_c1(rgb1.half()).clone()

        def _c4():
            torch.compiler.cudagraph_mark_step_begin()
            return stu_c4(rgb4.half()).clone()

        ms_tc_b1 = bench_gpu_callable(_c1, 1, warmup=8, iters=35)
        ms_tc_b4 = bench_gpu_callable(_c4, 4, warmup=8, iters=35)

        # TRT FP16 benchmark (reuse a0_b4_fp16.plan for A0_full to avoid duplicate 4-min build)
        if name == "A0_full" and (trt_dir / "a0_b4_fp16.plan").exists():
            plan_p = trt_dir / "a0_b4_fp16.plan"
        else:
            onnx_p = export_onnx_clean(stu_f32, (rgb4,), trt_dir / f"{name}_b4.onnx", ["rgb"], ["z_s"])
            plan_p, _ = build_trt_engine(onnx_p, trt_dir / f"{name}_b4_fp16.plan", precision="fp16")
        r_trt = TrtRunner(plan_p, ["rgb"], ["z_s"])
        ms_trt_b4 = bench_gpu_callable(lambda: r_trt(rgb4), 4, warmup=10, iters=45)

        prune_table.append({
            "name": name,
            "widths": widths,
            "params_M": round(meta["params"] / 1e6, 3),
            "flops_GF": round(gf, 2),
            "torch_compile_b1_ms": round(ms_tc_b1, 2),
            "torch_compile_b4_ms": round(ms_tc_b4, 2),
            "trt_fp16_b4_ms": round(ms_trt_b4, 2),
            "speedup_vs_torch48": round(48.01 / ms_tc_b4, 2),
            "speedup_trt_vs_48": round(48.01 / ms_trt_b4, 2),
        })

        (report_dir / "p1_prune_partial.json").write_text(json.dumps(prune_table, indent=2), encoding="utf-8")
        del stu_c1, stu_c4, stu_f16, stu_f32, stu, r_trt
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    out["identity_prune_max_err"] = round(id_err, 7)
    out["a0_prune_candidates"] = prune_table
    return out


def main() -> None:
    report_dir = OPT_OUT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    print("[P0] Running TRT FP16 + INT8 probes...", flush=True)
    p0_path = report_dir / "p0_trt_report.json"
    p0 = json.loads(p0_path.read_text(encoding="utf-8")) if p0_path.exists() else run_p0_trt_and_int8(report_dir)
    (report_dir / "p0_trt_report.json").write_text(json.dumps(p0, indent=2), encoding="utf-8")
    print("[P0 DONE]", json.dumps(p0, indent=2), flush=True)
    print("[P1] Running TAESD + A0 pruning design probes...", flush=True)
    p1 = run_p1_taesd_and_prune_design(report_dir)
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "p0": p0, "p1": p1}
    out_path = report_dir / "p0_p1_report.json"
    out_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("WROTE:", out_path, flush=True)
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
