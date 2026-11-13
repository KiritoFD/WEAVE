"""Benchmark + parity harness for the SD1.5 VAE decoder.

Backends
  B0  torch fp16 + channels_last + cudnn.benchmark (production `decode_latent`)
  B1  torch.compile(mode=reduce-overhead | max-autotune)
  B2  TensorRT engine (fixed shape, fp16 or fp32)
  B3  TensorRT engine replayed inside a torch CUDA Graph

Timing: CUDA events, warmup 20 / iters 50, report p50 / p90 / mean + img/s.
Parity: every backend is compared against B0 on the same *real* latents
  (max|Δ|, mean|Δ|, RMSE, PSNR, SSIM, NaN/Inf count, per-channel bias).

Usage (needs a free GPU; check `nvidia-smi --query-compute-apps` first):
  python tools/trt/bench_vae_decoder.py --sizes 64,512 --batches 1,8,32 \
      --precision fp16 --backends b0,b1,b2,b3 --compile-modes reduce-overhead,max-autotune
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools" / "trt"))

from paths import hf_cache_root, runs_root  # noqa: E402
from tools.legacy_utils.inference import load_vae, decode_latent  # noqa: E402

TRT_INPUT = "z"
TRT_OUTPUT = "image"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def timeit(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    ms_sorted = sorted(ms)
    return {
        "iters": iters,
        "mean_ms": round(statistics.fmean(ms), 2),
        "p50_ms": round(ms_sorted[len(ms) // 2], 2),
        "p90_ms": round(ms_sorted[int(0.9 * (len(ms) - 1))], 2),
        "min_ms": round(ms_sorted[0], 2),
        "max_ms": round(ms_sorted[-1], 2),
    }


def mem_stats() -> dict:
    """Peak torch allocation + whole-device usage (captures TRT's own allocator)."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "gpu_used_mb": round((total - free) / 2**20, 1),
    }


def compare(ref: torch.Tensor, got: torch.Tensor) -> dict:
    a = ref.detach().float().cpu().numpy()
    b = got.detach().float().cpu().numpy()
    diff = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    out = {
        "max_abs": float(diff.max()),
        "max_abs_255": float(diff.max() * 255.0),
        "mean_abs_255": float(diff.mean() * 255.0),
        "rmse": float(np.sqrt(mse)),
        "psnr_db": float(10 * np.log10(1.0 / max(mse, 1e-12))),
        "nan_count": int(np.isnan(b).sum()),
        "inf_count": int(np.isinf(b).sum()),
        "channel_mean_bias": [float((b[..., c] - a[..., c]).mean()) for c in range(min(3, a.shape[1] if a.ndim == 4 else 1))],
    }
    ssims = []
    for i in range(a.shape[0]):
        x = np.transpose(a[i], (1, 2, 0))
        y = np.transpose(b[i], (1, 2, 0))
        ssims.append(structural_similarity(x, y, channel_axis=2, data_range=1.0))
    out["ssim_mean"] = float(np.mean(ssims))
    return out


def make_batch(items: list[torch.Tensor], batch: int, lat: int) -> torch.Tensor:
    picked = items[:batch]
    while len(picked) < batch:
        picked = picked + picked
    picked = picked[:batch]
    return torch.stack([z if z.shape[-1] == lat else _resize(z, lat) for z in picked]).contiguous()


def _resize(z: torch.Tensor, lat: int) -> torch.Tensor:
    import torch.nn.functional as F
    return F.interpolate(z.unsqueeze(0), size=(lat, lat), mode="bicubic",
                         align_corners=False).squeeze(0)


# ---------------------------------------------------------------------------
# TensorRT backends
# ---------------------------------------------------------------------------
class TrtRunner:
    def __init__(self, engine_path: Path, device: str = "cuda"):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream

    def bind(self, z: torch.Tensor, out: torch.Tensor) -> None:
        self.ctx.set_tensor_address(TRT_INPUT, z.data_ptr())
        self.ctx.set_tensor_address(TRT_OUTPUT, out.data_ptr())

    def launch(self) -> None:
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,512")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--backends", default="b0,b1,b2,b3")
    ap.add_argument("--compile-modes", default="reduce-overhead,max-autotune")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--latents", default=str(runs_root("trt", "latents") / "real_latents.pt"))
    ap.add_argument("--onnx_dir", default=str(runs_root("trt", "onnx")))
    ap.add_argument("--engine_dir", default=str(runs_root("trt", "engines")))
    ap.add_argument("--out_dir", default=str(runs_root("trt", "bench")))
    ap.add_argument("--vae_cache", default=str(hf_cache_root()))
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    blob = torch.load(args.latents, map_location="cpu", weights_only=False)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    modes = [m.strip() for m in args.compile_modes.split(",") if m.strip()]

    vae = load_vae("cuda", model_id="ema", cache_dir=args.vae_cache, compile_decoder=False)
    vae = vae.to(device="cuda", dtype=torch.float16)

    from export_vae_decoder_onnx import DecoderExport
    mod = DecoderExport(vae).eval().cuda().half()

    results = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        lat = size // 8
        key = f"items_{size}"
        items = [v.float() for v in blob[key].values()] if key in blob else []
        if not items:
            print(f"skip size={size}: no latents under {key}")
            continue
        for batch in [int(x) for x in args.batches.split(",") if x.strip()]:
            z = make_batch(items, batch, lat).cuda()
            if args.precision == "fp16":
                z = z.half()
            row = {"size": size, "batch": batch, "precision": args.precision}

            # B0
            ref = None
            if "b0" in backends:
                def b0() -> torch.Tensor:
                    return decode_latent(vae, z, device="cuda")
                ref = b0()
                torch.cuda.reset_peak_memory_stats()
                row["b0"] = timeit(b0, args.warmup, args.iters)
                row["b0"].update(mem_stats())
                row["b0"]["img_per_s"] = round(batch / (row["b0"]["p50_ms"] / 1000), 1)

            # B1 per compile mode
            if "b1" in backends:
                for mode in modes:
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        t_compile = time.perf_counter()
                        compiled = torch.compile(mod, mode=mode)
                        fn = lambda: compiled(z)  # noqa: E731
                        got = fn()
                        torch.cuda.synchronize()
                        t = timeit(fn, args.warmup, args.iters)
                        t["compile_sec"] = round(time.perf_counter() - t_compile, 1)
                        t.update(mem_stats())
                        t["img_per_s"] = round(batch / (t["p50_ms"] / 1000), 1)
                        t["mode"] = mode
                        if ref is not None:
                            t["parity_vs_b0"] = compare(ref, got)
                        row[f"b1_{mode}"] = t
                    except Exception as exc:  # noqa: BLE001
                        row[f"b1_{mode}"] = {"error": repr(exc)[:300]}

            # B2 / B3
            if "b2" in backends or "b3" in backends:
                engine_path = Path(args.engine_dir) / f"vae_decoder_{size}_b{batch}_{args.precision}.engine"
                if not engine_path.exists():
                    row["b2"] = {"error": f"missing engine {engine_path}"}
                else:
                    try:
                        runner = TrtRunner(engine_path)
                        out_dtype = torch.float16 if args.precision == "fp16" else torch.float32
                        out = torch.zeros((batch, 3, size, size), device="cuda", dtype=out_dtype)
                        runner.bind(z.contiguous(), out)
                        runner.launch()
                        torch.cuda.synchronize()
                        if "b2" in backends:
                            torch.cuda.reset_peak_memory_stats()
                            t = timeit(runner.launch, args.warmup, args.iters)
                            t.update(mem_stats())
                            t["img_per_s"] = round(batch / (t["p50_ms"] / 1000), 1)
                            if ref is not None:
                                t["parity_vs_b0"] = compare(ref, out)
                            row["b2"] = t
                        if "b3" in backends:
                            graph = torch.cuda.CUDAGraph()
                            side = torch.cuda.Stream()
                            with torch.cuda.stream(side):
                                for _ in range(3):
                                    runner.launch()
                            torch.cuda.current_stream().wait_stream(side)
                            torch.cuda.synchronize()
                            with torch.cuda.graph(graph):
                                runner.launch()
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            t = timeit(graph.replay, args.warmup, args.iters)
                            t.update(mem_stats())
                            t["img_per_s"] = round(batch / (t["p50_ms"] / 1000), 1)
                            if ref is not None:
                                t["parity_vs_b0"] = compare(ref, out)
                            row["b3"] = t
                    except Exception as exc:  # noqa: BLE001
                        row["b2"] = {"error": repr(exc)[:300]}

            results.append(row)
            path = out_dir / f"bench_{size}_b{batch}_{args.precision}.json"
            path.write_text(json.dumps(row, indent=1), encoding="utf-8")
            print(json.dumps(row), flush=True)

    (out_dir / f"bench_all_{args.precision}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

