"""Optimized D5 latent pre-encode for local GPU (RTX 4070 Laptop 8GB).

Encoding-only path (decoder discarded). All compute in fp16.

The speedup strategy is exactly three things (not kernel tuning):
  * FUSE        : normalize+encoder captured as ONE cuda graph -> one "big op".
  * LAUNCH      : a single replay() per batch instead of per-layer launches.
  * NO_GRAD     : torch.inference_mode() -> no autograd graph, activations dropped.

Two data paths:
  * default  : stream JPEGs via cv2 (libjpeg-turbo) + pinned H2D.
  * --use_packed : read a single pre-packed uint8 HWC .bin via memmap
                   (zero decode, zero copy, no thread pool overhead).
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.inference import load_vae  # noqa: E402

# Clean pre-resized 512 JPEGs (fastest decode + no runtime resize).
D5_ROOT_CLEAN = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\train_512clean")
D5_ROOT_REMOTE = Path(r"I:\Github\Latent_Style\Dataset\distinct5_512\train_512clean")
D5_ROOT_RAW = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\train")
D5_ROOT_RAW_REMOTE = Path(r"I:\Github\Latent_Style\Dataset\distinct5_512\train")
# Auto-detect: use remote path if available, else local
D5_ROOT = D5_ROOT_REMOTE if D5_ROOT_REMOTE.is_dir() else (D5_ROOT_CLEAN if D5_ROOT_CLEAN.is_dir() else (D5_ROOT_RAW_REMOTE if D5_ROOT_RAW_REMOTE.is_dir() else D5_ROOT_RAW))
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SIZE = 512
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SCALE = 0.18215  # SD1.5 VAE scaling_factor

# Pre-packed uint8 HWC binary layout: [N, SIZE, SIZE, 3], row-major.
PACKED_BIN = _ROOT / "data" / "train_d5" / "images_hwc_uint8.bin"
PACKED_IDX = _ROOT / "data" / "train_d5" / "images_index.json"
NCHW_STRIDE = 3 * SIZE * SIZE


class EncoderOnlyVAE(nn.Module):
    """Minimal encode path: encoder -> quant_conv -> mean (mode)."""

    def __init__(self, encoder: nn.Module, quant_conv: nn.Module, scaling_factor: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean * self.scaling_factor


class FusedEncoder(nn.Module):
    """normalize(uint8 in [0,255]) + EncoderOnlyVAE in ONE module, for
    AOTInductor export. Compiled once -> shipped as a .pt2 that a consumer
    loads with torch._export.aot_load and runs with NO torch.compile at
    runtime (open-box). The .pt2 embeds the encoder weights too."""

    def __init__(self, enc: EncoderOnlyVAE) -> None:
        super().__init__()
        self.enc = enc

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.to(torch.float16) * (1.0 / 127.5) - 1.0
        x = x.contiguous(memory_format=torch.channels_last)
        return self.enc(x)


def export_onnx(enc: EncoderOnlyVAE, path: str, batch: int,
                device: torch.device, dtype: torch.dtype) -> None:
    """Export normalize+encoder to a FIXED-shape ONNX graph (batch fixed, no
    dynamic axes). This single .onnx file is the whole computation graph +
    weights; a consumer runs it with onnxruntime and NEVER calls torch.compile
    or rebuilds anything (open-box). uint8 [B,3,512,512] in -> fp16 latent out.
    """
    fused = FusedEncoder(enc).to(device=device, dtype=dtype).eval().requires_grad_(False)
    example = torch.zeros(batch, 3, SIZE, SIZE, dtype=torch.uint8, device=device)
    with torch.inference_mode():
        torch.onnx.export(
            fused, (example,), path,
            input_names=["images_uint8_nchw"], output_names=["latent"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )


def build_encoder_only(device: torch.device, dtype: torch.dtype, cache_dir: str) -> EncoderOnlyVAE:
    # encoder-only path: no attention in the VAE encoder, so xFormers is dead
    # weight (and can interfere with compile tracing) -> disable it.
    vae = load_vae(
        device=str(device), model_id="ema", cache_dir=cache_dir,
        enable_xformers=False, compile_decoder=False, compile_dtype=dtype,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)

    scale = float(getattr(vae.config, "scaling_factor", SCALE))
    enc = EncoderOnlyVAE(vae.encoder, vae.quant_conv, scale).to(
        device=device, dtype=dtype, memory_format=torch.channels_last,
    ).eval().requires_grad_(False)

    del vae.decoder, vae.post_quant_conv, vae
    torch.cuda.empty_cache()
    return enc


# ---------------------------------------------------------------------------
#  Image packing  (once): 5000 JPEGs -> one uint8 HWC .bin + index json
# ---------------------------------------------------------------------------
def _decode_bgr(path: Path) -> np.ndarray | None:
    """Read an image robustly even when the path has non-ASCII chars
    (cv2.imread on Windows fails on UTF-8 paths -> use numpy + imdecode)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def pack_images(cache_dir: str) -> None:
    PACKED_BIN.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    total = 0
    t0 = time.perf_counter()

    # write binary sequentially (safe for a single file handle); decode via
    # numpy+imdecode so non-ASCII filenames don't break cv2.imread on Windows.
    with open(PACKED_BIN, "wb") as fb:
        for style in STYLES:
            paths = sorted(p for p in (D5_ROOT / style).iterdir() if p.suffix.lower() in EXTS)
            for p in paths:
                img = _decode_bgr(p)
                if img is None:
                    raise RuntimeError(f"failed to decode {p}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (SIZE, SIZE):
                    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
                fb.write(np.ascontiguousarray(img).tobytes())
                records.append({"style": style, "file": p.name})
                total += 1

    PACKED_IDX.write_text(json.dumps({
        "root": str(D5_ROOT), "layout": "NHWC uint8", "dtype": "uint8",
        "n": total, "height": SIZE, "width": SIZE, "channels": 3,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"Packed {total} images -> {PACKED_BIN} ({PACKED_BIN.stat().st_size / 1e9:.2f} GB) in {time.perf_counter() - t0:.1f}s")
    print(f"Index -> {PACKED_IDX}")


def load_packed() -> tuple[np.memmap, list[dict], int]:
    meta = json.loads(PACKED_IDX.read_text(encoding="utf-8"))
    n = meta["n"]
    mmap = np.memmap(str(PACKED_BIN), dtype=np.uint8, mode="r",
                      shape=(n, SIZE, SIZE, 3))
    return mmap, meta["records"], n


# ---------------------------------------------------------------------------
#  Fast image loading
# ---------------------------------------------------------------------------
def load_one_uint8_chw(path: Path) -> np.ndarray:
    """Load via numpy+imdecode (robust to non-ASCII paths), uint8 CHW [0,255]."""
    img = _decode_bgr(path)
    if img is None:
        raise RuntimeError(f"failed to decode {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def prefetch_loader(paths: list[Path], batch_size: int, workers: int):
    """Background thread: decode JPEGs -> pinned NCHW uint8 tensor batches."""
    q: queue.Queue = queue.Queue(maxsize=4)

    def worker():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i in range(0, len(paths), batch_size):
                chunk = paths[i: i + batch_size]
                arrays = list(pool.map(load_one_uint8_chw, chunk))
                batch_uint8 = np.stack(arrays, axis=0)  # NCHW, contiguous
                q.put(torch.from_numpy(batch_uint8).pin_memory())
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def packed_prefetch_loader(records: list[dict], batch_size: int):
    """Background-thread prefetcher: copy memmap slices -> PINNED NCHW uint8
    tensors (queue). Pinned memory makes the later H2D truly async so the GPU
    never stalls on CPU prep. Mirrors prefetch_loader's design."""
    q: queue.Queue = queue.Queue(maxsize=4)
    n = len(records)

    def worker():
        for i in range(0, n, batch_size):
            b = min(batch_size, n - i)
            # pinned NCHW uint8 buffer (allocated once per slot)
            buf = torch.empty(b, 3, SIZE, SIZE, dtype=torch.uint8).pin_memory()
            src = np.ascontiguousarray(mmap_buf[i: i + b])          # (b,512,512,3) HWC
            buf.numpy()[:] = src.transpose(0, 3, 1, 2)              # HWC -> NCHW in pinned mem
            q.put(buf)
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


# ---------------------------------------------------------------------------
#  Encode loop
# ---------------------------------------------------------------------------
class CudaGraphEncoder:
    """Capture normalize+encoder into ONE cuda graph with a fixed static GPU
    input buffer. Each step: async H2D into the static buffer, then replay() ->
    zero kernel-launch overhead, zero per-step re-copy (the true 'big op')."""

    def __init__(self, forward_fn, batch: int, device: torch.device):
        self.batch = batch
        self.device = device
        # static uint8 input resident on GPU (filled via async copy each step)
        self.static_in = torch.empty(batch, 3, SIZE, SIZE, dtype=torch.uint8, device=device)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            # forward_fn takes the static uint8 buffer and returns the latent.
            # It may fuse normalize internally (e.g. an AOTInductor module).
            self.static_out = forward_fn(self.static_in)

    @torch.inference_mode()
    def run(self, cpu_pinned_uint8_nchw: torch.Tensor) -> torch.Tensor:
        # async H2D (pinned -> static GPU buffer); serializes on default stream
        self.static_in.copy_(cpu_pinned_uint8_nchw, non_blocking=True)
        self.g.replay()
        return self.static_out  # static memory; copy out before next replay


@torch.inference_mode()
def run_encode_graph(graph_enc: CudaGraphEncoder, it, collect: bool, device: torch.device):
    latents: list[torch.Tensor] = []
    count = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for cpu_buf in it:
        z = graph_enc.run(cpu_buf)            # z aliases static_out
        if collect:
            latents.append(z.detach().float().cpu())   # copy out before next replay
        count += z.shape[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    stack = torch.cat(latents, 0) if collect and latents else None
    del latents
    return elapsed, peak_vram, stack


class ORTEncoder:
    """Run the FIXED ONNX graph (normalize+encoder) via onnxruntime.
    NO torch.compile, NO per-run recompile. Providers, in priority order:
      * TensorRT EP (--onnx_trt): builds an fp16 engine ONCE, caches it to
        disk, and reuses it on every later run -> truly open-box + fastest.
      * CUDA EP (default): graph-optimized fixed-shape execution, zero build.
    The GPU input buffer is persistent and filled by async H2D each step
    (zero per-step host alloc), mirroring the cuda-graph path.
    """

    def __init__(self, onnx_path: str, batch: int, device: torch.device,
                 use_trt: bool = False, trt_cache: str = "") -> None:
        # onnxruntime-gpu needs the CUDA/cuDNN DLLs on the DLL search path.
        # torch already ships a matching set (cublasLt64_12.dll, cudnn64_9.dll,
        # ...) in torch/lib -> expose it so we don't require a system CUDA
        # toolkit install. (Must be done before importing onnxruntime.)
        import os
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            try:
                os.add_dll_directory(_torch_lib)
            except Exception:
                pass
        # TensorRT EP needs nvinfer_10.dll / nvinfer_plugin_10.dll from the
        # tensorrt_libs package (NOT on PATH by default) -> expose it too.
        try:
            import tensorrt_libs
            _tr_dir = os.path.dirname(tensorrt_libs.__file__)
            if os.path.isdir(_tr_dir):
                os.add_dll_directory(_tr_dir)
        except Exception:
            pass
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers: list = []
        if use_trt:
            providers.append(("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_fp16_enable": True,
                # Cap the tactic-search workspace so the ENGINE BUILD phase
                # (which tries dozens of kernels per op) cannot blow the 8GB
                # laptop VRAM. Inference-time VRAM stays low; only build is
                # bounded. Cache the built engine to disk for zero-rebuild reuse.
                "trt_max_workspace_size": 2_147_483_648,  # 2 GiB
                # Cache the built engine to disk for zero-rebuild reuse.
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache or str(Path(onnx_path).parent / "trt_cache"),
            }))
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        # persistent GPU input buffer, bound zero-copy via its device pointer
        self.static_in = torch.empty(batch, 3, SIZE, SIZE, dtype=torch.uint8, device=device)
        self.io = self.sess.io_binding()
        self.io.bind_input(name=self.in_name, device_type="cuda", device_id=0,
                           element_type=np.uint8, shape=tuple(self.static_in.shape),
                           buffer_ptr=self.static_in.data_ptr())
        # let ORT allocate the output on the GPU (no per-step D2H in the probe)
        self.io.bind_output(self.out_name, device_type="cuda", device_id=0)

    @torch.inference_mode()
    def run(self, cpu_pinned_uint8_nchw: torch.Tensor) -> None:
        self.static_in.copy_(cpu_pinned_uint8_nchw, non_blocking=True)
        self.sess.run_with_iobinding(self.io)

    def output_cpu_float(self) -> torch.Tensor:
        return torch.from_numpy(self.io.copy_outputs_to_cpu()[0]).float()


def run_encode_ort(ort_enc: ORTEncoder, it, collect: bool):
    latents: list[torch.Tensor] = []
    count = 0
    peak = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for cpu_buf in it:
        ort_enc.run(cpu_buf)
        if collect:
            latents.append(ort_enc.output_cpu_float())
        count += cpu_buf.shape[0]
        free, total = torch.cuda.mem_get_info()
        peak = max(peak, (total - free) / (1024 ** 2))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stack = torch.cat(latents, 0) if collect and latents else None
    del latents
    return elapsed, peak, stack


def save_latent_cache(stack: torch.Tensor, records: list[dict], out_root: Path,
                      batch_size: int, wall: float, peak: float, tag: str) -> int:
    packed = out_root / "packed"
    packed.mkdir(parents=True, exist_ok=True)
    counts: dict = {}
    for r in records:
        counts[r["style"]] = counts.get(r["style"], 0) + 1
    off = 0
    total = 0
    for style_id, style in enumerate(STYLES):
        c = counts.get(style, 0)
        sub = stack[off: off + c]
        files = [r["file"] for r in records[off: off + c]]
        torch.save({"schema": 1, "subdir": style, "count": c, "files": files, "latents": sub},
                   packed / f"{style_id:02d}_{style}.pt")
        off += c
        total += c
    meta = {"vae": tag, "source": "packed_bin", "total_samples": total,
            "image_size": SIZE, "batch_size": batch_size, "wall_sec": wall,
            "ms_per_img": 1000.0 * wall / max(total, 1), "peak_vram_mib": peak}
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return total


@torch.inference_mode()
def run_encode_stream(it, n: int, collect: bool, device: torch.device, enc):
    latents: list[torch.Tensor] = []
    count = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for item in it:
        if item.device == torch.device("cpu"):
            x = item.to(device, dtype=torch.float16, non_blocking=True)
            x = x.mul_(1.0 / 127.5).sub_(1.0).contiguous(memory_format=torch.channels_last)
        else:
            x = item  # already on GPU, fp16, channels_last, normalized
        z = enc(x)
        if collect:
            latents.append(z.detach().float().cpu())
        count += z.shape[0]
        del x, z

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    stack = torch.cat(latents, 0) if collect and latents else None
    del latents
    return elapsed, peak_vram, stack


def auto_find_batch(enc, device, dtype, vram_cap_bytes, max_bs=64, step=8) -> int:
    base = torch.cuda.memory_allocated(device)
    best = 1
    bs = step
    while bs <= max_bs:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            dummy = torch.empty(bs, 3, SIZE, SIZE, device=device, dtype=dtype).contiguous(
                memory_format=torch.channels_last)
            for _ in range(3):
                enc(dummy)
            torch.cuda.synchronize()
            act = torch.cuda.max_memory_allocated(device) - base
            if act <= vram_cap_bytes:
                best = bs
            else:
                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        bs += step
    torch.cuda.empty_cache()
    return best


# global memmap buffer ref (set in main when --use_packed)
mmap_buf: np.memmap | None = None


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    global mmap_buf
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=0, help="0 = auto (largest that fits --vram_cap_gb)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bench_n", type=int, default=256)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--output_dir", type=str, default=str(_ROOT / "data" / "train_d5" / ".latent_cache"))
    ap.add_argument("--vram_cap_gb", type=float, default=7.4)
    ap.add_argument("--no_cudnn_benchmark", action="store_true")
    ap.add_argument("--pack_images", action="store_true", help="build the single uint8 .bin (once)")
    ap.add_argument("--use_packed", action="store_true", help="read from pre-packed .bin via memmap")
    ap.add_argument("--sweep", type=str, default="",
                    help="comma batch sizes to probe, e.g. 4,8,16,24 (compiled, OOM-safe)")
    ap.add_argument("--graph", action="store_true",
                    help="manual CUDA graph (fused normalize+encoder, static buffer). "
                         "batch_size must divide the image count (e.g. 20/40 for 5000).")
    ap.add_argument("--graph_compile", action="store_true",
                    help="in --graph path: also torch.compile(default) fuse the encoder "
                         "before capturing the graph (faster; still no max-autotune).")
    ap.add_argument("--export_aoti", type=str, default="",
                    help="AOTInductor: compile+package fused encoder to this .pt2 (compile ONCE, "
                         "ship it). The batch size is fixed to --batch_size (or 8).")
    ap.add_argument("--load_aoti", type=str, default="",
                    help="AOTInductor: load this .pt2 and run with NO torch.compile. "
                         "Must be used with --graph and the SAME --batch_size it was exported with.")
    ap.add_argument("--export_onnx", type=str, default="",
                    help="Export fused (normalize+encoder) to a FIXED-shape .onnx graph "
                         "(batch = --batch_size or 8). Ship this single file; run it with "
                         "--load_onnx (onnxruntime, zero recompile).")
    ap.add_argument("--load_onnx", type=str, default="",
                    help="Run a .onnx graph via onnxruntime (NO torch.compile, zero recompile). "
                         "Requires --use_packed and the SAME --batch_size it was exported with.")
    ap.add_argument("--onnx_trt", action="store_true",
                    help="with --load_onnx: use the TensorRT EP (fp16, engine cached to disk "
                         "-> built once, reused). Falls back to CUDA EP if TRT is unavailable.")
    args = ap.parse_args()

    if args.pack_images:
        pack_images(str(_ROOT / "eval_cache" / "hf"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    vram_cap_bytes = args.vram_cap_gb * 1024 ** 3

    if device.type == "cuda" and not args.no_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}  total {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"image_root: {D5_ROOT}  (clean_512={D5_ROOT == D5_ROOT_CLEAN})  cap={args.vram_cap_gb:.1f} GB  use_packed={args.use_packed}")

    cache_dir = str(_ROOT / "eval_cache" / "hf")
    print("Building encoder-only FP16 VAE (decoder discarded)...")
    t_load = time.perf_counter()
    enc = build_encoder_only(device, dtype, cache_dir)
    print(f"  encoder loaded in {time.perf_counter() - t_load:.1f}s")

    # --- AOTInductor export: compile+package the fused encoder ONCE, ship the
    #     .pt2 so others run with NO torch.compile (open-box). ---
    if args.export_aoti:
        export_bs = args.batch_size if args.batch_size > 0 else 8
        print(f"AOTInductor: compiling+packaging fused (normalize+encoder) "
              f"-> {args.export_aoti}  (fixed batch={export_bs}) ...")
        fused = FusedEncoder(enc).to(device=device, dtype=dtype).eval().requires_grad_(False)
        example = (torch.empty(export_bs, 3, SIZE, SIZE, dtype=torch.uint8, device=device),)
        t_c = time.perf_counter()
        with torch.inference_mode():
            ep = torch.export.export(fused, example)
        try:
            pkg = torch._inductor.aoti_compile_and_package(ep, package_path=args.export_aoti)
            torch.cuda.synchronize()
            print(f"  done in {time.perf_counter() - t_c:.1f}s -> {pkg}")
            print("AOTI .pt2 (compiled .so inside). Others load with "
                  "--load_aoti -> ZERO recompile, open-box.")
        except Exception as e:  # AOTI needs a C++ compiler (MSVC cl.exe on Win)
            torch.export.save(ep, args.export_aoti)
            print(f"  AOTI C++ build skipped ({type(e).__name__}: C++ compiler not found).")
            print(f"  Saved the ExportedProgram (graph+weights) instead -> {args.export_aoti}")
            print("  Others: torch.export.load() then compile once (cached after).")
        return

    # --- ONNX export: one fixed-shape .onnx graph, open-box (no recompile) ---
    if args.export_onnx:
        export_bs = args.batch_size if args.batch_size > 0 else 8
        print(f"ONNX export: fused (normalize+encoder) -> {args.export_onnx}  "
              f"(fixed batch={export_bs}) ...")
        t_c = time.perf_counter()
        export_onnx(enc, args.export_onnx, export_bs, device, dtype)
        sz = Path(args.export_onnx).stat().st_size / 1e6 if Path(args.export_onnx).is_file() else 0.0
        print(f"  done in {time.perf_counter() - t_c:.1f}s  ({sz:.0f} MB)")
        print("Ship this .onnx; run it with --load_onnx (onnxruntime, ZERO recompile).")
        return

    # --- packed path setup ---
    if args.use_packed:
        if not PACKED_BIN.is_file():
            print("Packed .bin not found; run with --pack_images first.")
            return
        mmap_buf, records, total_imgs = load_packed()
        print(f"Loaded packed memmap: {total_imgs} images, {PACKED_BIN.stat().st_size / 1e9:.2f} GB")

    # --- ONNX runtime: run the fixed graph via onnxruntime (no torch.compile) ---
    if args.load_onnx:
        if not args.use_packed:
            print("--load_onnx currently requires --use_packed")
            return
        batch_size = args.batch_size if args.batch_size > 0 else 8
        oe = ORTEncoder(args.load_onnx, batch_size, device,
                        use_trt=args.onnx_trt, trt_cache="")
        print(f"onnxruntime provider in use: {oe.provider}")

        n_p = (args.bench_n // batch_size) * batch_size
        el, peak, _ = run_encode_ort(oe, packed_prefetch_loader(records[:n_p], batch_size),
                                     collect=False)
        ms = 1000.0 * el / n_p
        print(f"\n=== ONNX probe  n={n_p}  batch={batch_size}  provider={oe.provider} ===")
        print(f"elapsed={el:.2f}s  {ms:.2f} ms/img  {1000/ms:.1f} img/s")
        print(f"process VRAM peak: {peak:.0f} MiB")
        print(f">>> Projected 5000 images: {ms*5000/60000:.2f} min ({ms*5000/1000:.1f}s) <<<")

        if not args.full:
            print("\n(Pass --full to encode all 5000 and write packed cache.)")
            return

        if total_imgs % batch_size != 0:
            print(f"ERROR: --full --load_onnx needs batch_size={batch_size} to divide "
                  f"total={total_imgs} (try 20/25/40/50/100)")
            return
        out_root = Path(args.output_dir)
        t_all = time.perf_counter()
        el, peak_all, stack = run_encode_ort(oe, packed_prefetch_loader(records, batch_size),
                                             collect=True)
        wall = time.perf_counter() - t_all
        total = save_latent_cache(stack, records, out_root, batch_size, wall, peak_all,
                                  tag=f"sd15_ema_encoder_only_fp16_onnx_{oe.provider}")
        print("=" * 60)
        print(f"DONE  {total} images in {wall/60:.2f} min  ({1000.0*wall/max(total,1):.1f} ms/img)")
        print(f"provider={oe.provider}  process VRAM peak {peak_all:.0f} MiB  -> {out_root}")
        return

    # --- manual CUDA graph mode: fused normalize+encoder, static buffer ---
    if args.graph:
        if not args.use_packed:
            print("--graph currently requires --use_packed")
            return
        batch_size = args.batch_size if args.batch_size > 0 else 20
        if total_imgs % batch_size != 0:
            print(f"ERROR: --graph batch_size={batch_size} must divide total={total_imgs} "
                  f"(try 20/25/40/50/100)")
            return
        # Build the forward_fn (normalize + encoder) and capture it as ONE graph.
        # Two sources for forward_fn:
        #  * --load_aoti : AOTInductor .pt2 loaded with NO torch.compile. The
        #    normalize is already fused inside the packaged module. Open-box.
        #  * else         : eager encoder, optionally torch.compile(default) fused.
        #    cuDNN must be off (its conv allocates during capture ->
        #    DEVICE_ALLOCATION_FAILED); with cuDNN off the aten/cuBLAS path
        #    captures cleanly.
        if args.load_aoti:
            print(f"Loading AOTInductor package (no torch.compile at runtime): {args.load_aoti}")
            aoti_mod = torch._export.aot_load(args.load_aoti, "cuda")

            def forward_fn(x_uint8):
                return aoti_mod(x_uint8)
        else:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            if args.graph_compile:
                # Disable inductor's OWN cudagraphs: we capture the graph
                # manually below. Leaving it on double-wraps/nests graphs and
                # tanks throughput (per-step input re-copy). This is what made
                # the 43 ms/img result; without it you get ~130 ms/img.
                import torch._inductor.config as _ic
                _ic.triton.cudagraphs = False
                print("torch.compile(encoder) mode=default (fusion, no autotune) + manual cuda graph...")
                enc_c = torch.compile(enc, mode="default", fullgraph=True, dynamic=False)
                dummy = torch.empty(batch_size, 3, SIZE, SIZE, dtype=torch.uint8, device=device)
                _ = enc_c(dummy.to(dtype).mul_(1.0 / 127.5).sub_(1.0).contiguous(memory_format=torch.channels_last))
                torch.cuda.synchronize()
            else:
                enc_c = enc
                dummy = torch.empty(batch_size, 3, SIZE, SIZE, dtype=torch.uint8, device=device)
                _ = enc_c(dummy.to(dtype).mul_(1.0 / 127.5).sub_(1.0).contiguous(memory_format=torch.channels_last))
                torch.cuda.synchronize()
                print("Pure cuda graph (NO torch.compile). Capturing eager fp16 encoder...")

            def forward_fn(x_uint8):
                x = x_uint8.to(dtype).mul_(1.0 / 127.5).sub_(1.0).contiguous(memory_format=torch.channels_last)
                return enc_c(x)

        print("Capturing CUDA graph (static buffer, zero per-step copy)...")
        ge = CudaGraphEncoder(forward_fn, batch_size, device)

        # --- probe ---
        n_p = (args.bench_n // batch_size) * batch_size
        el, peak, _ = run_encode_graph(ge, packed_prefetch_loader(records[:n_p], batch_size),
                                       collect=False, device=device)
        ms = 1000.0 * el / n_p
        print(f"\n=== GRAPH probe  n={n_p}  batch={batch_size} ===")
        print(f"elapsed={el:.2f}s  {ms:.2f} ms/img  {1000/ms:.1f} img/s")
        print(f"peak VRAM: {peak:.0f} MiB  (cap {args.vram_cap_gb*1024:.0f})")
        print(f">>> Projected 5000 images: {ms*5000/60000:.2f} min ({ms*5000/1000:.1f}s) <<<")
        if peak > args.vram_cap_gb * 1024:
            print(f"WARNING: peak {peak:.0f} MiB exceeded cap")

        if not args.full:
            print("\n(Pass --full to encode all 5000 and write packed cache.)")
            return

        # --- full encode via graph ---
        out_root = Path(args.output_dir)
        packed = out_root / "packed"
        packed.mkdir(parents=True, exist_ok=True)
        t_all = time.perf_counter()
        total = 0
        el, peak_all, stack = run_encode_graph(ge, packed_prefetch_loader(records, batch_size),
                                               collect=True, device=device)
        counts = {}
        for r in records:
            counts[r["style"]] = counts.get(r["style"], 0) + 1
        off = 0
        for style_id, style in enumerate(STYLES):
            c = counts.get(style, 0)
            sub = stack[off: off + c]
            files = [r["file"] for r in records[off: off + c]]
            torch.save({"schema": 1, "subdir": style, "count": c, "files": files, "latents": sub},
                       packed / f"{style_id:02d}_{style}.pt")
            off += c
            total += c
        wall = time.perf_counter() - t_all
        meta = {"vae": "sd15_ema_encoder_only_fp16_graph", "source": "packed_bin",
                "total_samples": total, "image_size": SIZE, "batch_size": batch_size,
                "wall_sec": wall, "ms_per_img": 1000.0 * wall / max(total, 1),
                "peak_vram_mib": peak_all, "compile": True}
        (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("=" * 60)
        print(f"DONE  {total} images in {wall/60:.2f} min  ({meta['ms_per_img']:.1f} ms/img)")
        print(f"peak VRAM {peak_all:.0f} MiB  -> {out_root}")
        return

    # --- sweep mode: start small, time each batch segment, OOM-safe ---
    if args.sweep:
        batches = [int(x) for x in args.sweep.split(",") if x.strip()]
        print(f"\n##### SWEEP (compiled, max-autotune+cudagraphs) over batches={batches} #####")
        print(f"{'batch':>6} {'compile_s':>10} {'ms/img':>9} {'img/s':>8} {'peak_MiB':>10} {'vs_cap':>8}")
        enc_raw = enc
        best_bs, best_ms = None, float("inf")
        for bs in batches:
            torch.cuda.empty_cache()
            try:
                e = torch.compile(enc_raw, mode="max-autotune", fullgraph=True, dynamic=False)
                dummy = torch.empty(bs, 3, SIZE, SIZE, device=device, dtype=dtype).contiguous(
                    memory_format=torch.channels_last)
                t_c = time.perf_counter()
                for _ in range(5):
                    _ = e(dummy)
                torch.cuda.synchronize()
                t_comp = time.perf_counter() - t_c

                if args.use_packed:
                    n_p = min(args.bench_n, total_imgs)
                    it = packed_prefetch_loader(records[:n_p], bs)
                    el, peak, _ = run_encode_stream(it, n_p, collect=False, device=device, enc=e)
                else:
                    bpaths: list[Path] = []
                    for style in STYLES:
                        for p in sorted((D5_ROOT / style).iterdir()):
                            if p.suffix.lower() in EXTS:
                                bpaths.append(p)
                                if len(bpaths) >= args.bench_n:
                                    break
                        if len(bpaths) >= args.bench_n:
                            break
                    it = prefetch_loader(bpaths, bs, args.workers)
                    el, peak, _ = run_encode_stream(it, len(bpaths), collect=False, device=device, enc=e)
                ms = 1000.0 * el / n_p if args.use_packed else 1000.0 * el / len(bpaths)
                print(f"{bs:>6} {t_comp:>10.1f} {ms:>9.2f} {1000/ms:>8.1f} {peak:>10.0f} {peak-args.vram_cap_gb*1024:>+8.0f}")
                if peak <= args.vram_cap_gb * 1024 and ms < best_ms:
                    best_ms, best_bs = ms, bs
                del e
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{bs:>6}   OOM (skipped)")
                break
        print("##### SWEEP DONE #####")
        if best_bs is not None:
            print(f">>> best safe batch = {best_bs}  ->  {best_ms:.2f} ms/img  "
                  f"~{best_ms*5000/1000:.1f}s for 5000 (projected) <<<")
        else:
            print(">>> no batch fit under the VRAM cap <<<")
        return

    # --- decide batch ---
    if args.batch_size and args.batch_size > 0:
        batch_size = args.batch_size
    else:
        print("Auto-finding max batch under VRAM cap (uncompiled probe)...")
        batch_size = auto_find_batch(enc, device, dtype, vram_cap_bytes)
    print(f">>> batch_size = {batch_size} <<<")

    # --- compile ONCE ---
    if args.compile and device.type == "cuda":
        print("torch.compile(encoder) max-autotune + cudagraphs (one warmup ~30-60s)...")
        enc = torch.compile(enc, mode="max-autotune", fullgraph=True, dynamic=False)
        dummy = torch.empty(batch_size, 3, SIZE, SIZE, device=device, dtype=dtype).contiguous(
            memory_format=torch.channels_last)
        t_c = time.perf_counter()
        for _ in range(5):
            _ = enc(dummy)
        torch.cuda.synchronize()
        print(f"  compile+warmup done in {time.perf_counter() - t_c:.1f}s")

    # --- throughput probe ---
    if args.use_packed:
        probe_records = records[: args.bench_n]
        n_probe = len(probe_records)
        it_probe = packed_prefetch_loader(probe_records, batch_size)
        elapsed, peak, _ = run_encode_stream(it_probe, n_probe, collect=False, device=device, enc=enc)
    else:
        bench_paths: list[Path] = []
        for style in STYLES:
            for p in sorted((D5_ROOT / style).iterdir()):
                if p.suffix.lower() in EXTS:
                    bench_paths.append(p)
                    if len(bench_paths) >= args.bench_n:
                        break
            if len(bench_paths) >= args.bench_n:
                break
        n_probe = len(bench_paths)
        it_probe = prefetch_loader(bench_paths, batch_size, args.workers)
        elapsed, peak, _ = run_encode_stream(it_probe, n_probe, collect=False, device=device, enc=enc)

    ms = 1000.0 * elapsed / n_probe
    img_s = 1000.0 / ms
    proj_5k = ms * 5000 / 60000
    print(f"\n=== Throughput probe  n={n_probe}  batch={batch_size}  "
          f"{'packed' if args.use_packed else 'jpeg'} ===")
    print(f"elapsed={elapsed:.2f}s  {ms:.2f} ms/img  {img_s:.1f} img/s")
    print(f"peak VRAM (PyTorch alloc): {peak:.0f} MiB  (cap {args.vram_cap_gb * 1024:.0f})")
    print(f">>> Projected 5000 images: {proj_5k:.2f} min (={ms * 5000 / 1000:.1f}s) <<<")

    if peak > args.vram_cap_gb * 1024:
        print(f"WARNING: peak {peak:.0f} MiB exceeded cap; lower --batch_size or raise --vram_cap_gb")

    if not args.full:
        print("\n(Pass --full to encode all 5000 and write packed cache.)")
        return

    # --- full encode of all 5000 ---
    out_root = Path(args.output_dir)
    packed = out_root / "packed"
    packed.mkdir(parents=True, exist_ok=True)
    total = 0
    t_all = time.perf_counter()
    peak_all = peak

    if args.use_packed:
        elapsed_s, peak_s, stack = run_encode_stream(
            packed_prefetch_loader(records, batch_size), total_imgs,
            collect=True, device=device, enc=enc)
        peak_all = max(peak_all, peak_s)
        # split stack back into per-style .pt by index order
        counts = {}
        for r in records:
            counts[r["style"]] = counts.get(r["style"], 0) + 1
        offsets = {}
        off = 0
        for s in STYLES:
            offsets[s] = off
            off += counts.get(s, 0)
        for style_id, style in enumerate(STYLES):
            c = counts.get(style, 0)
            sub = stack[offsets[style]: offsets[style] + c]
            files = [r["file"] for r in records[offsets[style]: offsets[style] + c]]
            payload = {"schema": 1, "subdir": style, "count": c,
                       "files": files, "latents": sub}
            torch.save(payload, packed / f"{style_id:02d}_{style}.pt")
            total += c
            print(f"  {style}: {c} imgs -> {packed / (f'{style_id:02d}_{style}.pt')} {tuple(sub.shape)}")
        del stack
    else:
        for style_id, style in enumerate(STYLES):
            paths = sorted(p for p in (D5_ROOT / style).iterdir() if p.suffix.lower() in EXTS)
            elapsed_s, peak_s, stack = run_encode_stream(
                prefetch_loader(paths, batch_size, args.workers), len(paths),
                collect=True, device=device, enc=enc)
            peak_all = max(peak_all, peak_s)
            assert stack is not None
            payload = {"schema": 1, "subdir": style, "count": len(paths),
                       "files": [str(p.relative_to(D5_ROOT)) for p in paths],
                       "latents": stack}
            out_path = packed / f"{style_id:02d}_{style}.pt"
            torch.save(payload, out_path)
            total += len(paths)
            print(f"  {style}: {len(paths)} imgs  {1000 * elapsed_s / len(paths):.1f} ms/img  {tuple(stack.shape)}")
            del stack

    wall = time.perf_counter() - t_all
    meta = {
        "vae": "sd15_ema_encoder_only_fp16",
        "image_root": str(D5_ROOT),
        "source": "packed_bin" if args.use_packed else "jpeg_stream",
        "styles": STYLES,
        "total_samples": total,
        "image_size": SIZE,
        "batch_size": batch_size,
        "workers": args.workers,
        "wall_sec": wall,
        "ms_per_img": 1000.0 * wall / max(total, 1),
        "peak_vram_mib": peak_all,
        "compile": args.compile,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("=" * 60)
    print(f"DONE  {total} images in {wall / 60:.2f} min  ({meta['ms_per_img']:.1f} ms/img)")
    print(f"peak VRAM {peak_all:.0f} MiB  -> {out_root}")


if __name__ == "__main__":
    main()
