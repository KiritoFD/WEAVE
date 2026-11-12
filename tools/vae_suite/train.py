"""WEAVE Art-VAE Suite: Unified Training Script for A0, A1, A2.

Supports:
- Mode A0: Mixed Reconstruction (0.75 mean + 0.25 sample) + KL divergence.
- Mode A1: A0 + Multi-Scale Style CE (137 classes) with auxiliary classifier head.
- Mode A2: A0 + MoCo-style Supervised Contrastive Learning (137 classes, queue=4096).

Training Protocol:
- Architecture: Standard SD1.5 AutoencoderKL (83.65M params, latent 4x64x64).
- Precision: BF16 autocast, master weights in FP32.
- Optimizer: AdamW, Encoder LR 1e-5, Decoder LR 5e-6, quant_conv 1e-5.
- Gradient Checkpointing: enabled with use_reentrant=False.
- Gradient Clipping: 1.0.
- Checkpoints: saves regular checkpoints, latest.pt, and final exportable diffusers directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

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

from tools.vae_suite.dataset import WikiArtDataset, RawMemmapWikiArtDataset, load_wikiart_metadata, build_frozen_val2048
from tools.vae_suite.losses import (
    compute_kl_divergence,
    compute_mixed_reconstruction_loss,
    compute_moco_supcon_loss,
)
from tools.vae_suite.models import A1StyleClassifier, A2MoCoQueue
import lpips


class EMAModel:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data = s_p.data * self.decay + m_p.data * (1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def train_vae(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [Art-VAE Training] Mode: {args.mode} | Output: {out_dir} ===", flush=True)

    # 1. Load Calibration Manifest (or fallback to calibrated defaults)
    manifest_path = Path(args.calibration_manifest)
    if manifest_path.exists():
        calib = json.loads(manifest_path.read_text(encoding="utf-8"))
        beta = float(calib["gradient_calibration"]["calibrated_beta"])
        lambda_ce = float(calib["gradient_calibration"]["calibrated_lambda_ce"])
        lambda_con = float(calib["gradient_calibration"]["calibrated_lambda_con"])
        total_steps = args.steps if args.steps > 0 else int(calib.get("unified_training_steps_N", 2100))
    else:
        # Calibrated defaults from 32-batch gradient measurements on RTX 4090
        beta = 0.0001126
        lambda_ce = 0.00242
        lambda_con = 0.00352
        total_steps = args.steps if args.steps > 0 else 2100

    print(f"[Protocol] Unified Steps N = {total_steps} | beta = {beta:.7f} | lambda_ce = {lambda_ce:.5f} | lambda_con = {lambda_con:.5f}")

    # 2. Dataset & DataLoader (prefer ultra-fast zero-decode RAW memmap binary)
    if args.raw_manifest and Path(args.raw_manifest).exists():
        print(f"[Dataset] Loading high-speed Memory-Mapped RAW dataset from: {args.raw_manifest}", flush=True)
        train_ds = RawMemmapWikiArtDataset(Path(args.raw_manifest), is_training=True)
        num_styles = train_ds.manifest["num_styles"]
    else:
        img_roots = [
            Path(p) for p in args.image_roots.split(",") if Path(p).exists()
        ]
        records, style_to_id, artist_to_id, genre_to_id = load_wikiart_metadata(Path(args.csv_path), img_roots)
        num_styles = len(style_to_id)

        # Filter out 2048 validation set
        val_manifest = build_frozen_val2048(records, Path(args.val_manifest_path))
        val_filenames = {r["filename"] for r in val_manifest}
        train_records = [r for r in records if r["filename"] not in val_filenames and r.get("path")]
        train_ds = WikiArtDataset(train_records, image_size=512, is_training=True)

    # Fix random seed for exact reproducibility across A0, A1, A2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    accum = args.effective_batch // args.microbatch
    loader = DataLoader(
        train_ds,
        batch_size=args.microbatch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
    )
    print(f"[Dataset] Train size: {len(train_ds)} | Microbatch: {args.microbatch} | Accum: {accum} | Effective Batch: {args.effective_batch}")

    # 3. Model Initialization (strictly from args.vae_init_path)
    print(f"[Model] Initializing AutoencoderKL strictly from: {args.vae_init_path}", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_init_path).to(device)
    vae.train()
    vae.enable_gradient_checkpointing()

    compiled_vae = vae
    if not args.no_compile:
        print("[Compile] Compiling AutoencoderKL with dynamic=False (fixed 512x512)...", flush=True)
        compiled_vae = torch.compile(vae, mode="default", dynamic=False)

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    ema = EMAModel(vae, decay=0.999)

    # Auxiliary Heads
    a1_head = None
    a2_moco = None
    opt_param_groups = [
        {"params": list(vae.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(vae.decoder.parameters()), "lr": args.lr_decoder},
        {"params": list(vae.quant_conv.parameters()) + list(vae.post_quant_conv.parameters()), "lr": args.lr_encoder},
    ]

    if args.mode == "B1":
        print("[Mode B1] Freezing Encoder and training Decoder only...", flush=True)
        for p in vae.encoder.parameters():
            p.requires_grad = False
        for p in vae.quant_conv.parameters():
            p.requires_grad = False
        opt_param_groups = [
            {"params": list(vae.decoder.parameters()), "lr": args.lr_decoder},
            {"params": list(vae.post_quant_conv.parameters()), "lr": args.lr_decoder},
        ]
    elif args.mode == "A1":
        a1_head = A1StyleClassifier(in_features=296, num_styles=num_styles).to(device)
        opt_param_groups.append({"params": list(a1_head.parameters()), "lr": args.lr_encoder})
    elif args.mode == "A2":
        a2_moco = A2MoCoQueue(vae.encoder, queue_size=4096).to(device)
        opt_param_groups.append({"params": list(a2_moco.projector_q.parameters()), "lr": args.lr_encoder})
        # Pre-fill queue
        print("[A2 MoCo] Pre-populating queue with 512 samples...")
        init_loader = DataLoader(train_ds, batch_size=args.microbatch, shuffle=True)
        init_iter = iter(init_loader)
        for _ in range(512 // args.microbatch):
            b = next(init_iter)
            k = a2_moco.extract_key(b["image"].to(device))
            a2_moco.enqueue(k, b["style_id"].to(device), b["artist_id"].to(device))

    optimizer = torch.optim.AdamW(opt_param_groups, weight_decay=1e-4)

    # Cosine LR Scheduler with 2% Warmup
    warmup_steps = max(10, int(0.02 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Training Loop
    t_start = time.time()
    step = 0
    history = []

    print(f"\n=== [Starting Training Loop for {args.mode}] Target Steps: {total_steps} ===", flush=True)
    loader_iter = iter(loader)

    while step < total_steps:
        step += 1
        optimizer.zero_grad(set_to_none=True)
        step_l_rec = 0.0
        step_l_kl = 0.0
        step_l_aux = 0.0

        for a_idx in range(accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            x = batch["image"].to(device, non_blocking=True)
            st_ids = batch["style_id"].to(device, non_blocking=True)
            ar_ids = batch["artist_id"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                l_rec, l1, l_lp, posterior, _ = compute_mixed_reconstruction_loss(compiled_vae, x, lpips_fn)
                kl_elem, kl_val, kl_img, var_val = compute_kl_divergence(posterior)
                loss = (l_rec + beta * kl_elem) / accum

                if args.mode == "A1":
                    logits = a1_head(posterior.mean)
                    l_ce = nn.functional.cross_entropy(logits, st_ids)
                    loss = loss + (lambda_ce * l_ce) / accum
                    step_l_aux += float(l_ce.detach().item()) / accum
                elif args.mode == "A2":
                    q = a2_moco.extract_query(posterior.mean)
                    l_con, pos_rate = compute_moco_supcon_loss(
                        q, st_ids, ar_ids,
                        a2_moco.queue_feats, a2_moco.queue_styles, a2_moco.queue_artists,
                    )
                    loss = loss + (lambda_con * l_con) / accum
                    step_l_aux += float(l_con.detach().item()) / accum

            loss.backward()
            step_l_rec += float(l_rec.detach().item()) / accum
            step_l_kl += float(kl_val) / accum

        # Gradient Clipping & Optimizer Step
        grad_norm = float(torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0).item())
        optimizer.step()
        scheduler.step()
        ema.update(vae)

        if args.mode == "A2":
            a2_moco.update_momentum_encoder(vae.encoder)
            with torch.no_grad():
                k = a2_moco.extract_key(x)
                a2_moco.enqueue(k, st_ids, ar_ids)

        if step % 50 == 0 or step == total_steps:
            elapsed = time.time() - t_start
            sec_per_step = elapsed / step
            eta_h = (total_steps - step) * sec_per_step / 3600.0
            cur_lr = scheduler.get_last_lr()[0]
            peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)

            log_str = (
                f"[{args.mode}] Step {step:5d}/{total_steps} | "
                f"L_rec={step_l_rec:.4f} | KL={step_l_kl:.4f} | var={var_val:.4f} | "
                f"L_aux={step_l_aux:.4f} | g_norm={grad_norm:.2f} | lr={cur_lr:.2e} | "
                f"VRAM={peak_vram:.0f}M | Speed={sec_per_step:.3f}s/step | ETA={eta_h:.2f}h"
            )
            print(log_str, flush=True)

            history.append({
                "step": step,
                "l_rec": round(step_l_rec, 4),
                "l_kl": round(step_l_kl, 4),
                "posterior_var": round(var_val, 4),
                "l_aux": round(step_l_aux, 4),
                "grad_norm": round(grad_norm, 3),
                "lr": cur_lr,
                "elapsed_sec": round(elapsed, 1),
            })

        # Save regular checkpoints
        if step % args.save_every == 0 or step == total_steps:
            ckpt_path = out_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model": vae.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"[Saved] Checkpoint {ckpt_path}", flush=True)

    # 5. Export Final Standard Diffusers Model Directory
    final_dir = out_dir / "final_diffusers_vae"
    final_dir.mkdir(parents=True, exist_ok=True)
    # Load EMA weights into VAE for final export
    vae.load_state_dict(ema.state_dict())
    vae.save_pretrained(str(final_dir))
    print(f"\n[Export] Saved clean exportable AutoencoderKL to {final_dir}", flush=True)

    # Write training log
    (out_dir / "train_log.json").write_text(json.dumps({
        "mode": args.mode,
        "total_steps": total_steps,
        "total_duration_sec": round(time.time() - t_start, 1),
        "total_duration_hours": round((time.time() - t_start) / 3600.0, 2),
        "history": history,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A0", "A1", "A2", "B1"], required=True)
    ap.add_argument("--vae-init-path", default="/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema")
    ap.add_argument("--calibration-manifest", default="/root/Workspace/WEAVE_VAE/calibration_manifest.json")
    ap.add_argument("--csv-path", default="/root/Workspace/data/wikiart/all_data_info.csv")
    ap.add_argument("--image-roots", default="/root/Workspace/data/wikiart/train,/root/Workspace/data/wikiart/test")
    ap.add_argument("--val-manifest-path", default="/root/Workspace/WEAVE_VAE/val_2048_manifest.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--raw-manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--steps", type=int, default=2100, help="Unified steps N")
    ap.add_argument("--microbatch", type=int, default=6)
    ap.add_argument("--effective-batch", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-decoder", type=float, default=5e-6)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_vae(args)
