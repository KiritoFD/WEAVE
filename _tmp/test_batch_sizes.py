"""Test larger batch configurations on RTX 4090 for A0.
Tests:
- microbatch=4, effective_batch=32 (accum=8)
- microbatch=6, effective_batch=24 (accum=4)
- microbatch=8, effective_batch=32 (accum=4) with gradient checkpointing
"""
import time
import sys
from pathlib import Path
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

REPO_ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(REPO_ROOT))

from tools.vae_suite.dataset import WikiArtDataset, load_wikiart_metadata
from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
import lpips

device = "cuda" if torch.cuda.is_available() else "cpu"
vae_path = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
csv_path = Path("/root/Workspace/data/wikiart/all_data_info.csv")
img_roots = [Path("/root/Workspace/data/wikiart/test")]

records, style_to_id, _, _ = load_wikiart_metadata(csv_path, img_roots)
valid_records = [r for r in records if r.get("path") and Path(r["path"]).exists()]
dataset = WikiArtDataset(valid_records, image_size=512, is_training=True)
lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

configs = [
    (4, 16), # current baseline (microbatch=4, effective=16, accum=4)
    (4, 32), # microbatch=4, effective=32, accum=8
    (6, 24), # microbatch=6, effective=24, accum=4
    (8, 32), # microbatch=8, effective=32, accum=4
]

print("=== Testing Batch Configurations for A0 on RTX 4090 ===")

for mb, eb in configs:
    accum = eb // mb
    loader = DataLoader(dataset, batch_size=mb, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    loader_iter = iter(loader)

    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.train()
    vae.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-5)

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()

    try:
        # Warmup 2 updates
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                b = next(loader_iter)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(vae, b["image"].to(device), lpips_fn)
                    kl_elem, _, _, _ = compute_kl_divergence(posterior)
                    loss = (l_rec + 1e-4 * kl_elem) / accum
                loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_updates = 5

        for _ in range(n_updates):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                b = next(loader_iter)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(vae, b["image"].to(device), lpips_fn)
                    kl_elem, _, _, _ = compute_kl_divergence(posterior)
                    loss = (l_rec + 1e-4 * kl_elem) / accum
                loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
        sec_per_update = total_time / n_updates
        img_per_sec = (n_updates * eb) / total_time

        print(f"mb={mb:2d} | eff_batch={eb:2d} (accum={accum}) | Peak VRAM={peak_vram:6.0f} MiB ({peak_vram/24564*100:4.1f}%) | "
              f"Update Time={sec_per_update:5.2f}s | Throughput={img_per_sec:5.2f} img/s")
    except Exception as e:
        print(f"mb={mb:2d} | eff_batch={eb:2d} -> FAILED: {e}")
        torch.cuda.empty_cache()
