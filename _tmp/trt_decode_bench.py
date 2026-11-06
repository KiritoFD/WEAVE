"""TRT decode 单图耗时 + 全链 imgs/s 补测（b8/b16/b32）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "trt"))

import paths
import trt_vae_decoder
from utils.inference import load_vae, decode_latent

device = "cuda"
vae = load_vae(device, model_id="ema", cache_dir=str(paths.eval_cache_root() / "hf"), compile_decoder=False)
trt = trt_vae_decoder.TrtVaeDecoder(ROOT / "runs/trt/engines", vae, precision="fp16", device=device)

for db in (8, 16, 32):
    print(f"[decode] batch={db} engine_exists={trt.usable(db, 512)}", flush=True)
    z = torch.randn(db, 4, 64, 64, device=device, dtype=torch.float16)
    for _ in range(3):
        trt.decode(z)
    torch.cuda.synchronize()
    n = 20
    t0 = time.time()
    for _ in range(n):
        trt.decode(z)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / n
    print(f"[decode] batch={db} total_ms={ms:.2f} ms_per_img={ms / db:.2f} imgs_per_s={db / (ms / 1000):.2f}", flush=True)

# torch fp16 decode 参考（batch 8）
z = torch.randn(8, 4, 64, 64, device=device, dtype=torch.float16)
for _ in range(2):
    decode_latent(vae, z, device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    decode_latent(vae, z, device)
torch.cuda.synchronize()
ms = (time.time() - t0) * 1000 / 10
print(f"[decode] torch_fp16 batch=8 total_ms={ms:.2f} ms_per_img={ms / 8:.2f} imgs_per_s={8 / (ms / 1000):.2f}", flush=True)
print("TRT_DECODE_BENCH_DONE")
