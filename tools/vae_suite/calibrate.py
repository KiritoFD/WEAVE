"""WEAVE Art-VAE Suite: Gradient Calibration & 100-Step Throughput Benchmark.

Executes:
1. Measures raw gradient norms for L_rec, L_KL, L_CE, and L_SupCon across 32 batches.
2. Calibrates:
   - beta so that ||grad(beta * L_KL)|| / ||grad(L_rec)|| in [1%, 5%]
   - lambda_ce so that ||grad(lambda_ce * L_CE)|| / ||grad(L_rec)|| ~= 15%
   - lambda_con so that ||grad(lambda_con * L_SupCon)|| / ||grad(L_rec)|| ~= 12%
3. Runs 100 steps each of A0, A1, A2 to measure wall-clock throughput and peak VRAM.
4. Computes unified update steps N to fit within the 4-hour budget:
   N_A0 = N_A1 = N_A2, B_effective_A0 = B_effective_A1 = B_effective_A2.
5. Saves configuration to calibration_manifest.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import WikiArtDataset, load_wikiart_metadata, build_frozen_val2048
from tools.vae_suite.losses import (
    compute_kl_divergence,
    compute_mixed_reconstruction_loss,
    compute_moco_supcon_loss,
)
from tools.vae_suite.models import A1StyleClassifier, A2MoCoQueue
import lpips


def get_gradient_norm(named_params, module_filter: str = "encoder") -> float:
    """Calculate total L2 gradient norm for specific module parameters."""
    grads = []
    for name, p in named_params:
        if module_filter in name and p.grad is not None:
            grads.append(p.grad.detach().flatten().float())
    if not grads:
        return 0.0
    return float(torch.cat(grads).norm(2).item())


def run_gradient_calibration(
    vae_path: str,
    dataset: WikiArtDataset,
    num_styles: int,
    device: str = "cuda",
    num_batches: int = 32,
    microbatch: int = 4,
) -> dict:
    print(f"\n=== [Step 1] Running Gradient Calibration on {num_batches} Batches ===")
    loader = DataLoader(dataset, batch_size=microbatch, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.train()
    vae.enable_gradient_checkpointing()

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    a1_head = A1StyleClassifier(in_features=296, num_styles=num_styles).to(device)
    a2_moco = A2MoCoQueue(vae.encoder, queue_size=512).to(device)

    # Pre-fill a small queue for SupCon
    print("[Calibration] Warmup filling A2 queue with initial batches...")
    for _ in range(8):
        b = next(loader_iter)
        x = b["image"].to(device)
        with torch.no_grad():
            k = a2_moco.extract_key(x)
            a2_moco.enqueue(k, b["style_id"].to(device), b["artist_id"].to(device))

    rec_norms = []
    kl_norms = []
    ce_norms = []
    con_norms = []

    for idx in range(num_batches):
        batch = next(loader_iter)
        x = batch["image"].to(device)
        st_ids = batch["style_id"].to(device)
        ar_ids = batch["artist_id"].to(device)

        # 1. Measure L_rec gradient norm
        vae.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_rec, _, _, posterior, z_mixed = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
        l_rec.backward(retain_graph=True)
        g_rec = get_gradient_norm(vae.named_parameters(), "encoder")
        rec_norms.append(g_rec)

        # 2. Measure L_KL gradient norm
        vae.zero_grad(set_to_none=True)
        kl_elem, _, _, _ = compute_kl_divergence(posterior)
        kl_elem.backward(retain_graph=True)
        g_kl = get_gradient_norm(vae.named_parameters(), "encoder")
        kl_norms.append(g_kl)

        # 3. Measure L_CE gradient norm
        vae.zero_grad(set_to_none=True)
        a1_head.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = a1_head(posterior.mean)
            l_ce = nn.functional.cross_entropy(logits, st_ids)
        l_ce.backward(retain_graph=True)
        g_ce = get_gradient_norm(vae.named_parameters(), "encoder")
        ce_norms.append(g_ce)

        # 4. Measure L_SupCon gradient norm
        vae.zero_grad(set_to_none=True)
        a2_moco.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            q = a2_moco.extract_query(posterior.mean)
            l_con, _ = compute_moco_supcon_loss(
                q, st_ids, ar_ids,
                a2_moco.queue_feats, a2_moco.queue_styles, a2_moco.queue_artists,
            )
        if l_con.requires_grad:
            l_con.backward()
            g_con = get_gradient_norm(vae.named_parameters(), "encoder")
            con_norms.append(g_con)

    mean_g_rec = float(np.mean(rec_norms))
    mean_g_kl = float(np.mean(kl_norms))
    mean_g_ce = float(np.mean(ce_norms))
    mean_g_con = float(np.mean(con_norms)) if con_norms else 1.0

    # Calibrate coefficients:
    # Target 1: ||beta * grad_KL|| / ||grad_rec|| ~= 0.025 (midpoint of [1%, 5%])
    calibrated_beta = float(0.025 * mean_g_rec / max(mean_g_kl, 1e-6))

    # Target 2: ||lambda_ce * grad_CE|| / ||grad_rec|| ~= 0.15
    calibrated_lambda_ce = float(0.15 * mean_g_rec / max(mean_g_ce, 1e-6))

    # Target 3: ||lambda_con * grad_SupCon|| / ||grad_rec|| ~= 0.12
    calibrated_lambda_con = float(0.12 * mean_g_rec / max(mean_g_con, 1e-6))

    calib_res = {
        "mean_grad_norm_rec": round(mean_g_rec, 4),
        "mean_grad_norm_kl": round(mean_g_kl, 4),
        "mean_grad_norm_ce": round(mean_g_ce, 4),
        "mean_grad_norm_con": round(mean_g_con, 4),
        "calibrated_beta": round(calibrated_beta, 7),
        "calibrated_lambda_ce": round(calibrated_lambda_ce, 5),
        "calibrated_lambda_con": round(calibrated_lambda_con, 5),
        "target_kl_gradient_ratio": "2.5% (within 1% - 5%)",
        "target_ce_gradient_ratio": "15.0%",
        "target_con_gradient_ratio": "12.0%",
    }
    print("[Calibration] Results:", json.dumps(calib_res, indent=2))
    return calib_res


def benchmark_mode_100_steps(
    vae_path: str,
    dataset: WikiArtDataset,
    num_styles: int,
    mode: str,
    calib: dict,
    device: str = "cuda",
    microbatch: int = 4,
    effective_batch: int = 16,
) -> dict:
    print(f"\n--- [Benchmark] Testing 100 Steps Throughput for Mode: {mode} ---")
    accum = effective_batch // microbatch
    loader = DataLoader(dataset, batch_size=microbatch, shuffle=True, drop_last=True)
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
        # pre-populate queue
        for _ in range(16):
            b = next(loader_iter)
            k = a2_moco.extract_key(b["image"].to(device))
            a2_moco.enqueue(k, b["style_id"].to(device), b["artist_id"].to(device))

    optimizer = torch.optim.AdamW(opt_params, lr=1e-5)

    beta = calib["calibrated_beta"]
    l_ce_w = calib["calibrated_lambda_ce"]
    l_con_w = calib["calibrated_lambda_con"]

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()

    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)
        for a_idx in range(accum):
            batch = next(loader_iter)
            x = batch["image"].to(device)
            st_ids = batch["style_id"].to(device)
            ar_ids = batch["artist_id"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                l_rec, l1, l_lp, posterior, _ = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
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
            # Enqueue batch keys
            with torch.no_grad():
                k = a2_moco.extract_key(x)
                a2_moco.enqueue(k, st_ids, ar_ids)

    torch.cuda.synchronize()
    total_sec = time.perf_counter() - t0
    peak_vram_mib = torch.cuda.max_memory_allocated(0) / (1024**2)
    sec_per_step = total_sec / 100.0

    print(f"[{mode} 100-step Benchmark] Time: {total_sec:.2f}s ({sec_per_step:.3f} s/update) | Peak VRAM: {peak_vram_mib:.0f} MiB")
    return {
        "mode": mode,
        "sec_per_step": round(sec_per_step, 4),
        "peak_vram_mib": round(peak_vram_mib, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", default="/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema")
    ap.add_argument("--csv-path", default="/root/Workspace/data/wikiart/all_data_info.csv")
    ap.add_argument("--dest", default="/root/Workspace/WEAVE_VAE")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse metadata & build dataset
    img_roots = [
        Path("/root/Workspace/data/wikiart/train"),
        Path("/root/Workspace/data/wikiart/test"),
        Path("/root/Workspace/WEAVE_VAE/datasets/distinct5_512"),
    ]
    records, style_to_id, artist_to_id, genre_to_id = load_wikiart_metadata(Path(args.csv_path), img_roots)
    print(f"[Dataset] Parsed {len(records)} records. Identified {len(style_to_id)} styles, {len(artist_to_id)} artists, {len(genre_to_id)} genres.")

    # Save style dictionary
    (dest_dir / "style_to_id.json").write_text(json.dumps(style_to_id, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build frozen 2048 validation set
    val_manifest = build_frozen_val2048(records, dest_dir / "val_2048_manifest.json")

    # Filter out val records from training pool
    val_filenames = {r["filename"] for r in val_manifest}
    train_records = [r for r in records if r["filename"] not in val_filenames and r.get("path")]
    print(f"[Dataset] Training pool has {len(train_records)} resolvable images (strictly disjoint from 2048 val).")

    train_ds = WikiArtDataset(train_records, image_size=512, is_training=True)

    # 2. Gradient Calibration
    calib = run_gradient_calibration(args.vae_path, train_ds, num_styles=len(style_to_id), device=device, num_batches=32)

    # 3. 100-step Benchmark on A0, A1, A2
    benchmarks = {}
    for m in ("A0", "A1", "A2"):
        b_res = benchmark_mode_100_steps(args.vae_path, train_ds, num_styles=len(style_to_id), mode=m, calib=calib, device=device)
        benchmarks[m] = b_res

    # 4. Compute unified update steps N for 4-hour budget
    # 4 hours = 14400 seconds. Add 10% safety margin -> budget = 13000 seconds.
    max_sec_per_step = max(b["sec_per_step"] for b in benchmarks.values())
    budget_sec = 13000.0
    unified_N = int(budget_sec / max_sec_per_step)
    # Round down to nearest 100
    unified_N = (unified_N // 100) * 100

    print(f"\n=== [Unified Training Budget Alignment] ===")
    print(f"Max sec/step across (A0, A1, A2): {max_sec_per_step:.3f} s")
    print(f"Unified update steps N: {unified_N} steps (strictly identical for A0, A1, A2)")
    print(f"Estimated training duration per model: {unified_N * max_sec_per_step / 3600.0:.2f} hours")

    final_manifest = {
        "vae_init_path": args.vae_path,
        "num_styles": len(style_to_id),
        "num_training_images": len(train_records),
        "num_val_images": len(val_manifest),
        "effective_batch_size": 16,
        "microbatch_size": 4,
        "gradient_accumulation_steps": 4,
        "gradient_calibration": calib,
        "benchmarks_100_steps": benchmarks,
        "unified_training_steps_N": unified_N,
        "estimated_duration_hours": round(unified_N * max_sec_per_step / 3600.0, 2),
    }

    manifest_file = dest_dir / "calibration_manifest.json"
    manifest_file.write_text(json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved] Wrote final calibration manifest to {manifest_file}")


if __name__ == "__main__":
    main()
