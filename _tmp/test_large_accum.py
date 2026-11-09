"""Test large accumulation batch sizes with microbatch=6 on RTX 4090:
- mb=6, eff=96 (accum=16)
- mb=6, eff=120 (accum=20)
- mb=6, eff=144 (accum=24)
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import sys
from pathlib import Path
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

REPO_ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(REPO_ROOT))

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.losses import (
    compute_mixed_reconstruction_loss,
    compute_kl_divergence,
    compute_moco_supcon_loss,
)
from tools.vae_suite.models import A2MoCoQueue
import lpips

device = "cuda" if torch.cuda.is_available() else "cpu"
vae_path = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
manifest_path = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")

dataset = RawMemmapWikiArtDataset(manifest_path, is_training=True)
lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

print("=== Testing microbatch=6 with 100+ Total Batch Accumulation on RTX 4090 ===")

configs = [
    (6, 96, 16),   # mb=6, accum=16, effective=96
    (6, 120, 20),  # mb=6, accum=20, effective=120
    (6, 144, 24),  # mb=6, accum=24, effective=144
]

for mb, eb, accum in configs:
    loader = DataLoader(dataset, batch_size=mb, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    loader_iter = iter(loader)

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.train()
    vae.enable_gradient_checkpointing()
    compiled_vae = torch.compile(vae, mode="default", dynamic=False)

    a2_moco = A2MoCoQueue(vae.encoder, queue_size=8192).to(device)
    opt_params = list(vae.parameters()) + list(a2_moco.projector_q.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=1e-5)

    try:
        # Warmup
        for _ in range(1):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                b = next(loader_iter)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(compiled_vae, b["image"].to(device), lpips_fn)
                    loss = (l_rec + 1e-4 * posterior.kl().mean()) / accum
                loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_updates = 2

        for _ in range(n_updates):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                b = next(loader_iter)
                x = b["image"].to(device, non_blocking=True)
                st = b["style_id"].to(device, non_blocking=True)
                ar = b["artist_id"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(compiled_vae, x, lpips_fn)
                    q = a2_moco.extract_query(posterior.mean)
                    l_con, _ = compute_moco_supcon_loss(
                        q, st, ar, a2_moco.queue_feats, a2_moco.queue_styles, a2_moco.queue_artists
                    )
                    loss = (l_rec + 1e-4 * posterior.kl().mean() + 0.0035 * l_con) / accum
                loss.backward()
            optimizer.step()
            a2_moco.update_momentum_encoder(vae.encoder)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
        sec_update = dt / n_updates
        speed = (n_updates * eb) / dt

        print(f"SUCCESS: mb={mb} | eff_batch={eb:3d} (accum={accum:2d}) | Peak VRAM={peak_vram:6.0f} MiB ({peak_vram/24564*100:4.1f}%) | "
              f"Time={sec_update:5.2f}s/step | Speed={speed:5.2f} img/s")
    except Exception as e:
        print(f"FAILED mb={mb}, eb={eb}: {e}")
        torch.cuda.empty_cache()
