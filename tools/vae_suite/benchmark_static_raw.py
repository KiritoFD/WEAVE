"""WEAVE Art-VAE Suite: Precision Benchmark with Zero-Decode Raw RAM Dataset & Static-Shape torch.compile.

Key Optimizations:
1. Data source: /root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json (75 GB memory-mapped uint8).
   - Zero CPU decode, zero resize, instantaneous RAM slice.
2. Static Shape torch.compile:
   - dynamic=False: lets Inductor emit optimal fused Triton kernels for fixed 512x512 / 64x64 grids.
3. Tests user-selected configuration:
   - microbatch = 6, effective_batch = 24 (accum = 4).
4. Separated timing:
   - Step 1: Compile / Warmup time.
   - Steps 2..20: Steady-state launch time (sec/update and img/s throughput).
   - Peak VRAM tracking on RTX 4090.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Inductor cache settings
INDUCTOR_CACHE = "/root/Workspace/.inductor_cache"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = INDUCTOR_CACHE
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.losses import (
    compute_kl_divergence,
    compute_mixed_reconstruction_loss,
    compute_moco_supcon_loss,
)
from tools.vae_suite.models import A1StyleClassifier, A2MoCoQueue
import lpips


def run_benchmark_for_mode(
    vae_path: str,
    dataset: RawMemmapWikiArtDataset,
    num_styles: int,
    mode: str,
    device: str = "cuda",
    microbatch: int = 6,
    effective_batch: int = 24,
    use_compile: bool = True,
) -> dict:
    print(f"\n=======================================================", flush=True)
    print(f"  Benchmark Mode: {mode} (mb={microbatch}, eff={effective_batch}, compile={use_compile})", flush=True)
    print(f"=======================================================", flush=True)

    accum = effective_batch // microbatch
    loader = DataLoader(
        dataset,
        batch_size=microbatch,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    loader_iter = iter(loader)

    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.train()
    vae.enable_gradient_checkpointing()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    a1_head = None
    a2_moco = None
    opt_params = list(vae.parameters())

    if mode == "A1":
        a1_head = A1StyleClassifier(in_features=296, num_styles=num_styles).to(device)
        opt_params += list(a1_head.parameters())
    elif mode == "A2":
        a2_moco = A2MoCoQueue(vae.encoder, queue_size=4096).to(device)
        opt_params += list(a2_moco.projector_q.parameters())
        # Warmup queue with 120 samples
        for _ in range(120 // microbatch):
            b = next(loader_iter)
            k = a2_moco.extract_key(b["image"].to(device))
            a2_moco.enqueue(k, b["style_id"].to(device), b["artist_id"].to(device))

    # Static Shape torch.compile
    compiled_vae = vae
    if use_compile:
        print(f"[Compile] Compiling AutoencoderKL with dynamic=False (fixed 512x512)...", flush=True)
        compiled_vae = torch.compile(vae, mode="default", dynamic=False)

    optimizer = torch.optim.AdamW(opt_params, lr=1e-5)

    # Standard calibrated loss weights
    beta = 0.0001126
    l_ce_w = 0.00242
    l_con_w = 0.00352

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Step 1: Compile & First Launch Step (JIT kernel compilation)
    # -------------------------------------------------------------
    print("[Timing] Running Step 1 (Triggering Static JIT compilation / warmup)...", flush=True)
    torch.cuda.synchronize()
    t_compile_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    for a_idx in range(accum):
        batch = next(loader_iter)
        x = batch["image"].to(device, non_blocking=True)
        st_ids = batch["style_id"].to(device, non_blocking=True)
        ar_ids = batch["artist_id"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(compiled_vae, x, lpips_fn)
            kl_elem, _, _, _ = compute_kl_divergence(posterior)
            loss = (l_rec + beta * kl_elem) / accum

            if mode == "A1":
                logits = a1_head(posterior.mean)
                l_ce = nn.functional.cross_entropy(logits, st_ids)
                loss = loss + (l_ce_w * l_ce) / accum
            elif mode == "A2":
                q = a2_moco.extract_query(posterior.mean)
                l_con, _ = compute_moco_supcon_loss(
                    q, st_ids, ar_ids,
                    a2_moco.queue_feats, a2_moco.queue_styles, a2_moco.queue_artists,
                )
                loss = loss + (l_con_w * l_con) / accum

        loss.backward()

    torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
    optimizer.step()

    if mode == "A2":
        a2_moco.update_momentum_encoder(vae.encoder)
        with torch.no_grad():
            k = a2_moco.extract_key(x)
            a2_moco.enqueue(k, st_ids, ar_ids)

    torch.cuda.synchronize()
    compile_warmup_sec = time.perf_counter() - t_compile_start
    print(f"  -> Step 1 (Compile + Warmup Launch): {compile_warmup_sec:.3f} s", flush=True)

    # -------------------------------------------------------------
    # Steps 2..20: Pure Steady-State Execution (19 update steps)
    # -------------------------------------------------------------
    n_steady_steps = 19
    step_times = []
    print(f"[Timing] Measuring Steady-State Launch over Steps 2..20 ({n_steady_steps} updates)...", flush=True)

    for step in range(2, 21):
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        for a_idx in range(accum):
            batch = next(loader_iter)
            x = batch["image"].to(device, non_blocking=True)
            st_ids = batch["style_id"].to(device, non_blocking=True)
            ar_ids = batch["artist_id"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                l_rec, _, _, posterior, _ = compute_mixed_reconstruction_loss(compiled_vae, x, lpips_fn)
                kl_elem, _, _, _ = compute_kl_divergence(posterior)
                loss = (l_rec + beta * kl_elem) / accum

                if mode == "A1":
                    logits = a1_head(posterior.mean)
                    l_ce = nn.functional.cross_entropy(logits, st_ids)
                    loss = loss + (l_ce_w * l_ce) / accum
                elif mode == "A2":
                    q = a2_moco.extract_query(posterior.mean)
                    l_con, _ = compute_moco_supcon_loss(
                        q, st_ids, ar_ids,
                        a2_moco.queue_feats, a2_moco.queue_styles, a2_moco.queue_artists,
                    )
                    loss = loss + (l_con_w * l_con) / accum

            loss.backward()

        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()

        if mode == "A2":
            a2_moco.update_momentum_encoder(vae.encoder)
            with torch.no_grad():
                k = a2_moco.extract_key(x)
                a2_moco.enqueue(k, st_ids, ar_ids)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t_step_start
        step_times.append(dt)

    peak_vram_mib = torch.cuda.max_memory_allocated(0) / (1024**2)
    mean_step_sec = float(np.mean(step_times))
    std_step_sec = float(np.std(step_times))
    img_per_sec = float(effective_batch / mean_step_sec)

    res = {
        "mode": mode,
        "use_compile": use_compile,
        "dynamic_shapes": False,
        "microbatch": microbatch,
        "effective_batch": effective_batch,
        "accum_steps": accum,
        "compile_warmup_sec": round(compile_warmup_sec, 3),
        "steady_sec_per_update": round(mean_step_sec, 4),
        "steady_std_sec": round(std_step_sec, 4),
        "throughput_img_per_sec": round(img_per_sec, 2),
        "peak_vram_mib": round(peak_vram_mib, 1),
        "vram_utilization_pct": round(peak_vram_mib / 24564 * 100.0, 1),
    }

    print(f"[{mode} Results] Compile/Warmup: {res['compile_warmup_sec']:.2f}s | "
          f"Steady Step Time: {res['steady_sec_per_update']:.3f} ± {res['steady_std_sec']:.3f} s/update | "
          f"Throughput: {res['throughput_img_per_sec']:.2f} img/s | Peak VRAM: {res['peak_vram_mib']:.0f} MiB ({res['vram_utilization_pct']}%)",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", default="/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema")
    ap.add_argument("--manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--microbatch", type=int, default=6)
    ap.add_argument("--effective-batch", type=int, default=24)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Static-Shape Compile & RAM Preload Benchmark on RTX 4090 ===", flush=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    dataset = RawMemmapWikiArtDataset(manifest_path, is_training=True)
    num_styles = dataset.manifest["num_styles"]
    print(f"[Dataset] Loaded zero-decode RAM dataset. Total train images: {len(dataset)}, Styles: {num_styles}", flush=True)

    results = {}
    for m in ("A0", "A1", "A2"):
        b_res = run_benchmark_for_mode(
            args.vae_path, dataset, num_styles, m,
            device=device, microbatch=args.microbatch, effective_batch=args.effective_batch, use_compile=True
        )
        results[m] = b_res

    out_file = Path("/root/Workspace/WEAVE_VAE/benchmark_static_raw_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] Wrote benchmark results to {out_file}", flush=True)


if __name__ == "__main__":
    main()
