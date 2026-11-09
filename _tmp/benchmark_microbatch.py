"""Microbatch & VRAM Benchmark on RTX 4090:
Tests AutoencoderKL at 512x512 with BF16 autocast and gradient checkpointing
for microbatch in [2, 4, 8].
"""
import time
import sys
from pathlib import Path
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

REPO_ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(REPO_ROOT))

from tools.vae_suite.dataset import WikiArtDataset, load_wikiart_metadata
from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
import lpips

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0)}")

# Check BF16 support
print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")

vae_path = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
csv_path = Path("/root/Workspace/data/wikiart/all_data_info.csv")
img_roots = [Path("/root/Workspace/data/wikiart/test")]

records, style_to_id, _, _ = load_wikiart_metadata(csv_path, img_roots)
valid_records = [r for r in records if r.get("path") and Path(r["path"]).exists()]
print(f"Available test images: {len(valid_records)}")

dataset = WikiArtDataset(valid_records, image_size=512, is_training=True)
lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

vae = AutoencoderKL.from_pretrained(vae_path).to(device)
vae.train()
vae.enable_gradient_checkpointing()
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-5)

for mb in [2, 4, 8]:
    loader = DataLoader(dataset, batch_size=mb, shuffle=True, drop_last=True)
    loader_iter = iter(loader)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()

    # Warmup 5 steps
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        b = next(loader_iter)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(vae, b["image"].to(device), lpips_fn)
            kl_elem, _, _, _ = compute_kl_divergence(posterior)
            loss = l_rec + 1e-4 * kl_elem
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_test = 20

    for _ in range(n_test):
        optimizer.zero_grad(set_to_none=True)
        b = next(loader_iter)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(vae, b["image"].to(device), lpips_fn)
            kl_elem, _, _, _ = compute_kl_divergence(posterior)
            loss = l_rec + 1e-4 * kl_elem
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    total_sec = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
    img_per_sec = (n_test * mb) / total_sec
    sec_per_step = total_sec / n_test

    print(f"Microbatch = {mb:2d} | Peak VRAM = {peak_vram:6.0f} MiB ({peak_vram/24564*100:.1f}%) | "
          f"Throughput = {img_per_sec:5.2f} img/s | Step Time = {sec_per_step*1000:6.1f} ms")
