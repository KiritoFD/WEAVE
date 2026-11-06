"""Decode-only throughput/numerics bench (eval-side, no training)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(_p))

import trt_vae_decoder  # noqa: E402
from utils.inference import load_vae, decode_latent  # noqa: E402

N = 64
device = "cuda"
torch.manual_seed(0)
vae = load_vae(device, model_id="ema", cache_dir="/mnt/i/Latent_Style/eval_cache/hf", compile_decoder=False)
trt = trt_vae_decoder.TrtVaeDecoder(ROOT / "runs" / "trt" / "engines", vae, precision="fp16", device=device)
z = torch.randn(N, 4, 64, 64, device=device, dtype=torch.float16) * 0.5

ref = None
for b in (8, 16, 32, 64):
    outs = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, N, b):
            outs.append(trt.decode(z[s:s + b]))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    img = torch.cat(outs, 0).float()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    if ref is None:
        ref = img
        diff = 0.0
    else:
        diff = float((img - ref).abs().max())
    print(f"[dec] trt b={b} {dt:.3f}s {dt/N*1000:.1f} ms/img peak_alloc={peak:.0f}mib maxdiff_vs_b8={diff:.3e}",
          flush=True)

for b in (16, 32, 64):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        torch.cuda.synchronize()
        for s in range(0, N, b):
            _ = decode_latent(vae, z[s:s + b], device)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"[dec] torch b={b} {dt:.3f}s {dt/N*1000:.1f} ms/img peak_alloc={peak:.0f}mib", flush=True)

print(json.dumps({"engines": [p.name for p in (ROOT / 'runs' / 'trt' / 'engines').glob('vae_decoder_512_b*_fp16.engine')]}))
