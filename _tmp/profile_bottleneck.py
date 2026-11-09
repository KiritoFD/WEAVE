"""Profile computation vs IO/CPU bottlenecks on RTX 4090 with 251GB RAM.
Measures:
1. CPU JPEG decode + BICUBIC 512 Resize latency per batch of 6.
2. In-memory uint8 (pre-resized 512) load latency per batch of 6.
3. Pure GPU computation latency (Forward + Backward + Optimizer) per batch of 6.
"""
import time
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL

REPO_ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(REPO_ROOT))

from tools.vae_suite.dataset import WikiArtDataset, load_wikiart_metadata
from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
import lpips
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"
vae_path = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
csv_path = Path("/root/Workspace/data/wikiart/all_data_info.csv")
img_roots = [Path("/root/Workspace/data/wikiart/test")]

records, _, _, _ = load_wikiart_metadata(csv_path, img_roots)
valid_records = [r for r in records if r.get("path") and Path(r["path"]).exists()]

# --------------------------------------------------------------------
# 1. Profile CPU Disk Read + Decode + 512 Resize (Batch of 6)
# --------------------------------------------------------------------
tf = T.Compose([
    T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

sample_paths = [r["path"] for r in valid_records[:60]]
times_disk_cpu = []
cached_uint8 = []

for s in range(0, 60, 6):
    chunk = sample_paths[s : s + 6]
    t0 = time.perf_counter()
    tensors = []
    for p in chunk:
        im = Image.open(p).convert("RGB")
        im_resized = im.resize((512, 512), Image.BICUBIC)
        cached_uint8.append(np.array(im_resized, dtype=np.uint8))
        tensors.append(tf(im))
    x_batch = torch.stack(tensors)
    times_disk_cpu.append((time.perf_counter() - t0) * 1000.0)

mean_disk_cpu_ms = float(np.mean(times_disk_cpu))

# --------------------------------------------------------------------
# 2. Profile In-Memory (RAM Cache) Load (Batch of 6)
# --------------------------------------------------------------------
cached_uint8_arr = np.stack(cached_uint8[:60]) # (60, 512, 512, 3) in RAM
times_ram = []
for s in range(0, 60, 6):
    t0 = time.perf_counter()
    chunk_np = cached_uint8_arr[s : s + 6] # slice from RAM
    t_batch = torch.from_numpy(chunk_np).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    times_ram.append((time.perf_counter() - t0) * 1000.0)

mean_ram_ms = float(np.mean(times_ram))

# --------------------------------------------------------------------
# 3. Profile Pure GPU Computation (Batch of 6)
# --------------------------------------------------------------------
vae = AutoencoderKL.from_pretrained(vae_path).to(device)
vae.train()
vae.enable_gradient_checkpointing()
lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-5)

x_gpu = torch.randn(6, 3, 512, 512, device=device, dtype=torch.float32)

# Warmup GPU
for _ in range(5):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(vae, x_gpu, lpips_fn)
        loss = l_rec + 1e-4 * posterior.kl().mean()
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
times_gpu = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(vae, x_gpu, lpips_fn)
        loss = l_rec + 1e-4 * posterior.kl().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    times_gpu.append((time.perf_counter() - t0) * 1000.0)

mean_gpu_ms = float(np.mean(times_gpu))

print("=== PROFILING RESULTS: COMPUTATION vs IO/CPU (Batch of 6, 512x512) ===")
print(f"1. Disk Read + CPU JPEG Decode + BICUBIC Resize : {mean_disk_cpu_ms:6.1f} ms / batch-6")
print(f"2. RAM Cache Slice (Zero-decode, Pre-resized)   : {mean_ram_ms:6.1f} ms / batch-6")
print(f"3. Pure GPU Forward + Backward + Optimizer      : {mean_gpu_ms:6.1f} ms / batch-6")
print(f"-------------------------------------------------------------------------")
print(f"Ratio: CPU/Disk is {mean_disk_cpu_ms / mean_gpu_ms * 100.0:.1f}% of Pure GPU computation time!")
