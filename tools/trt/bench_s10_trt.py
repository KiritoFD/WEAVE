"""Benchmark fixed-shape TensorRT VAE-decoder engines (batch 8 vs 12).

Loads engines built for the paper baseline (official sd-vae-ft-ema decoder,
runs/trt/engines/) and for the new SD0-S10 decoders (runs/trt/engines_s10new/),
runs each with random latents and reports CUDA-event ms/img.

Usage (3060, Windows python):
  python tools/trt/bench_s10_trt.py --batches 8,12 --iters 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
TRT = _ROOT / "runs/trt"
ENG_BASE = TRT / "engines"          # official vae_decoder_<size>_b<batch>_fp16.engine
ENG_NEW = TRT / "engines_s10new"    # s10new_<variant>_decoder_<size>_b<batch>_fp16.engine
SIZE = 512
LAT = SIZE // 8


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        ss[i].record()
        fn()
        ee[i].record()
    torch.cuda.synchronize()
    return sum(a.elapsed_time(b) for a, b in zip(ss, ee)) / iters


def run_engine(engine_path: Path, batch: int, iters: int, warmup: int) -> dict:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(str(engine_path), "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to load {engine_path}")
    ctx = engine.create_execution_context()
    name = engine.get_tensor_name(0)
    # fixed-shape engines: min==opt==max==batch
    ctx.set_input_shape(name, (batch, 4, LAT, LAT))
    dt = trt.nptype(engine.get_tensor_dtype(name))
    tdt = torch.from_numpy(np.empty(0, dtype=dt)).dtype
    z = (torch.randn(batch, 4, LAT, LAT, device="cuda", dtype=torch.float32) / 0.18215).to(tdt)
    out_name = engine.get_tensor_name(1)
    out_shape = tuple(engine.get_tensor_shape(out_name))
    out_dt = trt.nptype(engine.get_tensor_dtype(out_name))
    out = torch.empty(out_shape, device="cuda",
                      dtype=torch.from_numpy(np.empty(0, dtype=out_dt)).dtype)
    ctx.set_tensor_address(name, int(z.data_ptr()))
    ctx.set_tensor_address(out_name, int(out.data_ptr()))
    bindings = [int(z.data_ptr()), int(out.data_ptr())]
    if hasattr(ctx, "execute_v3"):
        ms = timed(lambda: ctx.execute_v3(bindings), iters, warmup)
    else:
        ms = timed(lambda: ctx.execute_v2(bindings), iters, warmup)
    return {"engine": str(engine_path), "batch": batch,
            "ms_per_batch": round(ms, 3), "ms_per_img": round(ms / batch, 3),
            "img_per_s": round(batch / ms * 1000, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="8,12")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    rows = []
    for b in [int(x) for x in args.batches.split(",") if x.strip()]:
        # official baseline engine (existing file, cached build)
        p = ENG_BASE / f"vae_decoder_{SIZE}_b{b}_fp16.engine"
        if p.exists():
            rows.append({"variant": "official", **run_engine(p, b, args.iters, args.warmup)})
            print(json.dumps(rows[-1]), flush=True)
        for v in ("old_s10", "c0", "c1"):
            p = ENG_NEW / f"s10new_{v}_decoder_{SIZE}_b{b}_fp16.engine"
            if not p.exists():
                print(f"skip {v} b{b}: missing {p}", flush=True)
                continue
            rows.append({"variant": v, **run_engine(p, b, args.iters, args.warmup)})
            print(json.dumps(rows[-1]), flush=True)

    out = TRT / "engines_s10new" / f"bench_s10_trt.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
