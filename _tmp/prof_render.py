"""Stage timing for pgt_render pipeline (throughput diagnosis only)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(_p))

import eval_fast as ef
import paths
import trt_vae_decoder
from utils.inference import load_vae, LGTInference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--cache-dir", default=str(paths.eval_cache_root()))
    ap.add_argument("--engine-dir", default=str(ROOT / "runs" / "trt" / "engines"))
    ap.add_argument("--config-override", default=str(ROOT / "inference.json"))
    args = ap.parse_args()

    args.source_latent_cache = True
    args.vae_model = "ema"
    args.encode_batch = 1
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    t0 = time.perf_counter()
    test_images, requests, style_meta = ef.build_protocol(ROOT / "data" / "test", max_src_samples=30, ref_pool="first")
    lgt = LGTInference(model_path=args.checkpoint, device=device, num_steps=args.num_steps,
                       step_size=1.0, style_strength=1.0, residual_scale=1.0,
                       config_override_path=args.config_override)
    vae = load_vae(device, model_id="ema", cache_dir=str(Path(args.cache_dir) / "hf"), compile_decoder=False)
    trt = trt_vae_decoder.TrtVaeDecoder(args.engine_dir, vae, precision="fp16", device=device)
    print(f"[prof] load models {time.perf_counter()-t0:.1f}s", flush=True)

    for holder in ("model_cfg", "bridge_cfg"):
        cfg = getattr(lgt.model, holder, None)
        if cfg is not None and hasattr(cfg, "endpoint_adain_scale"):
            cfg.endpoint_adain_scale = 0.0

    tim = ef.Timer(sync_cuda=True)
    source_paths = sorted({r["src_path"] for r in requests})[: args.n]
    src_lat = ef.build_source_latents(lgt, vae, source_paths, device, args, tim)

    ns = argparse.Namespace(
        disable_clip=False, disable_lpips=False, disable_dino=False, test_dir=str(ROOT / "data" / "test"),
        clip_model="openai/clip-vit-base-patch32", clip_local_only=True,
        clip_cache_dir=str(Path(args.cache_dir) / "hf" / "hub"), clip_skip_processor=False,
        amp=False, lpips_net="alex", lpips_chunk=32, dino_model="facebook/dinov2-small",
        dino_cache_dir=str(Path(args.cache_dir) / "hf"), allow_network=False,
        ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=args.image_size,
        metric_batch=32, ref_batch=32, only_clip_style=True, clip_style_idt_baseline=0.0, clip_vectors=False,
    )
    engine = ef.MetricEngine(ns, test_images, style_meta, device)
    inner = ef.Timer(sync_cuda=True)
    tim.start("metric_load")
    engine.load(inner)
    engine.prepare(inner, source_paths)
    tim.stop("metric_load")
    print(f"[prof] metric engine ready {time.perf_counter()-t0:.1f}s", flush=True)

    rows = [req for req in requests if req["src_path"] in set(source_paths)][: args.n]
    req_lat = src_lat[: len(rows)]
    tgt = rows[0]["tgt_style_id"]
    latent0 = req_lat.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        latent0 = lgt.inversion(latent0)
    style_lat_src = src_lat[0:1].to(device=device, dtype=torch.float32)
    tgt_ids = torch.full((len(rows),), tgt, dtype=torch.long, device=device)

    for B in (64, 128):
        n = min(B, len(rows))
        x0 = latent0[:n]
        sty = style_lat_src.expand(n, -1, -1, -1).contiguous()
        t = time.perf_counter()
        with torch.inference_mode():
            gen = lgt.generation_with_target_latent(x0, tgt_ids[:n], num_steps=args.num_steps,
                                                    target_style_latent={"style_latent_tensor": sty})
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"[prof] ODE batch={n} {dt:.3f}s = {dt/n*1000:.1f} ms/img", flush=True)

    gen = gen.float()
    t = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, gen.shape[0], 8):
            chunk = gen[s:s + 8]
            k = chunk.shape[0]
            if k < 8:
                chunk = torch.cat([chunk, chunk[-1:].expand(8 - k, -1, -1, -1)], 0)
            out = trt.decode(chunk)[:k]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    print(f"[prof] decode batch=8 n={gen.shape[0]} {dt:.3f}s = {dt/gen.shape[0]*1000:.1f} ms/img", flush=True)

    gen01 = out.float().clamp(0, 1)
    for mb in (32, 128):
        ns.metric_batch = mb
        rows64 = rows[-out.shape[0]:]
        t = time.perf_counter()
        engine.batch_metrics(gen01, rows64, ef.Timer(sync_cuda=True))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"[prof] metrics batch={mb} n={len(rows64)} {dt:.3f}s = {dt/len(rows64)*1000:.1f} ms/img", flush=True)

    print("[prof] summary " + json.dumps(tim.as_dict(), default=float), flush=True)


if __name__ == "__main__":
    main()
