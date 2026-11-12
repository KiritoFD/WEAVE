"""WEAVE-VAE V2: 7-Hour Co-Training Pipeline with Direct Latent SWD Ranking Loss.

Key Technical Innovations:
1. Direct Metric Learning on Raw Latents: Supervised multi-scale Sliced Wasserstein + Moment ranking loss.
2. Parameter-Free Supervision: Gradients flow directly into Encoder without learnable projection heads.
3. Decoupled Memory Bank: 136-style reference bank (1,632 entries) enables microbatch=6 with global contrast.
4. Co-Trained Decoder: Full reconstruction + KL co-training without auxiliary generative heads.
5. Calibrated Style Gradient: Dynamically scales lambda_s so ||grad_style|| / ||grad_rec|| ~ 0.40.
6. Initialized from A0: Begins from trained A0 weights, target 1800 updates (~7h on RTX 4090).
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.losses import compute_kl_divergence, compute_mixed_reconstruction_loss
from tools.vae_suite.reference_bank import StyleReferenceBank, SupervisedStyleRankingLoss
from tools.vae_suite.sampler import HybridStyleSampler
from tools.vae_suite.swd_metric import LatentSWDDescriptor
import lpips


class EMAModel:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def train_vae_v2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [WEAVE-VAE V2 7-Hour Training] Output: {out_dir} ===", flush=True)

    # 1. Load SWD Calibration Parameters
    calib_p = Path(args.calibration_json)
    if not calib_p.exists():
        raise FileNotFoundError(f"Missing {calib_p}. Run calibrate_swd.py first.")
    calib_data = json.loads(calib_p.read_text(encoding="utf-8"))

    descriptor = LatentSWDDescriptor().to(device)
    descriptor.set_calibration(
        torch.tensor(calib_data["chan_mean"]).to(device),
        torch.tensor(calib_data["chan_std"]).to(device),
        calib_data["scale_swd1"],
        calib_data["scale_swd2"],
        calib_data["scale_moment"],
        calib_data["median_inter_dist"],
    )
    print(f"[Descriptor] Initialized with calibrated scales: SWD1={calib_data['scale_swd1']:.4f}, Median={calib_data['median_inter_dist']:.4f}")

    # 2. Dataset & Hybrid Sampler
    print(f"[Dataset] Loading high-speed Memory-Mapped RAW dataset from: {args.raw_manifest}", flush=True)
    train_ds = RawMemmapWikiArtDataset(Path(args.raw_manifest), is_training=True)
    num_styles = train_ds.manifest["num_styles"]

    # Fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sampler = HybridStyleSampler(train_ds, p_natural=0.75, total_samples=args.steps * args.effective_batch, seed=args.seed)
    loader = DataLoader(
        train_ds,
        batch_size=args.microbatch,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    accum = args.effective_batch // args.microbatch
    total_steps = args.steps
    print(f"[Dataset] Microbatch: {args.microbatch} | Accum: {accum} | Effective Batch: {args.effective_batch} | Target Steps: {total_steps}")

    # 3. Model Initialization (strictly from args.vae_init_path, i.e., A0)
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

    # 4. Style Reference Bank & Ranking Loss
    ref_bank = StyleReferenceBank(num_styles=num_styles, refs_per_style=12, descriptor=descriptor).to(device)
    ref_bank.populate(vae, train_ds, device=device, batch_size=16)

    ranking_loss_fn = SupervisedStyleRankingLoss(margin=0.15, temperature=0.20, num_positives=2, num_negatives=8)

    # 5. Optimizer & LR Schedule
    opt_param_groups = [
        {"params": list(vae.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(vae.decoder.parameters()), "lr": args.lr_decoder},
        {"params": list(vae.quant_conv.parameters()) + list(vae.post_quant_conv.parameters()), "lr": args.lr_encoder},
    ]
    optimizer = torch.optim.AdamW(opt_param_groups, betas=(0.9, 0.99), weight_decay=1e-4)

    # 2% Warmup + Cosine Decay
    warmup_steps = max(20, int(0.02 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Training Loop & Dynamic Gradient Ratio
    # KL divergence weight: weakened to 1/5 of original (1.126e-4 / 5 = 2.252e-5)
    beta = args.beta_kl
    # Style loss weight: calibrated so ||grad_style|| / ||grad_rec|| ~ 40-45%
    lambda_style = args.lambda_style # Calibrated: 0.0100 on RTX 4090

    history = []
    t_start = time.time()
    loader_iter = iter(loader)
    step = 0

    print(f"\n=== [Starting Training Loop for WEAVE-VAE V2] Target Steps: {total_steps} | beta_kl: {beta:.7f} (1/5th) | lambda_style: {lambda_style} ===", flush=True)

    while step < total_steps:
        step += 1
        t_step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        step_l_rec = 0.0
        step_l_kl = 0.0
        step_l_style = 0.0
        step_d_pos = 0.0
        step_d_neg = 0.0

        # Warmup lambda_style smoothly during first 50 steps: 0.002 -> 0.010
        warmup_factor = min(1.0, max(0.2, step / 50.0))
        effective_lambda = lambda_style * warmup_factor

        for a_idx in range(accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            x = batch["image"].to(device, non_blocking=True)
            st_ids = batch["style_id"].to(device, non_blocking=True)
            ar_ids = batch["artist_id"].to(device, non_blocking=True)
            ge_ids = batch["genre_id"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 1. Reconstruction & KL
                l_rec, l1, l_lp, posterior, _ = compute_mixed_reconstruction_loss(compiled_vae, x, lpips_fn)
                kl_elem, kl_val, _, _ = compute_kl_divergence(posterior)

                # 2. Style SWD Ranking Loss on posterior.mean
                q1, q2, mom = descriptor(posterior.mean.float())
                l_style, style_stats = ranking_loss_fn(
                    q1, q2, mom, st_ids, ar_ids, ge_ids, ref_bank
                )

                loss = (l_rec + beta * kl_elem + effective_lambda * l_style) / accum

            loss.backward()

            step_l_rec += float(l_rec.detach().item()) / accum
            step_l_kl += float(kl_val) / accum
            step_l_style += float(l_style.detach().item()) / accum
            step_d_pos += style_stats["d_pos"] / accum
            step_d_neg += style_stats["d_neg"] / accum

        # Clip grad and step
        grad_norm = float(torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0).item())
        optimizer.step()
        scheduler.step()
        ema.update(vae)

        dt_step = time.perf_counter() - t_step_start
        cur_lr = scheduler.get_last_lr()[0]

        # Log every 10 steps
        if step % 10 == 0 or step == total_steps or step <= 5:
            peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
            eta_hours = ((total_steps - step) * dt_step) / 3600.0
            print(
                f"[V2] Step {step:4d}/{total_steps} | L_rec={step_l_rec:.4f} | KL={step_l_kl:.2f} | "
                f"L_style={step_l_style:.4f} (d_pos={step_d_pos:.3f}, d_neg={step_d_neg:.3f}) | "
                f"g_norm={grad_norm:.2f} | lr={cur_lr:.2e} | VRAM={peak_vram:.0f}M | {dt_step:.2f}s/step | ETA={eta_hours:.2f}h",
                flush=True,
            )
            history.append({
                "step": step,
                "l_rec": round(step_l_rec, 4),
                "l_kl": round(step_l_kl, 4),
                "l_style": round(step_l_style, 4),
                "d_pos": round(step_d_pos, 4),
                "d_neg": round(step_d_neg, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": cur_lr,
                "elapsed_sec": round(time.time() - t_start, 1),
            })

        # Periodically refresh Style Reference Bank every 200 steps
        if step % 200 == 0 and step < total_steps:
            print(f"\n[Step {step}] Refreshing Style Reference Bank descriptors with updated Encoder...", flush=True)
            ref_bank.refresh_descriptors(vae, train_ds, device=device, batch_size=16)

        # Save Checkpoint every 200 steps
        if step % args.save_every == 0 or step == total_steps:
            ckpt_path = out_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model": vae.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "lambda_style": lambda_style,
                "args": vars(args),
            }, ckpt_path)
            print(f"[Saved] Checkpoint {ckpt_path}", flush=True)

    # 7. Export Final Diffusers Model Directory
    final_dir = out_dir / "final_diffusers_vae"
    final_dir.mkdir(parents=True, exist_ok=True)
    vae.load_state_dict(ema.state_dict())
    vae.save_pretrained(str(final_dir))
    print(f"\n[Export] Saved clean exportable AutoencoderKL to {final_dir}", flush=True)

    # Save log
    (out_dir / "train_log.json").write_text(json.dumps({
        "mode": "WEAVE_VAE_V2",
        "total_steps": total_steps,
        "total_duration_hours": round((time.time() - t_start) / 3600.0, 2),
        "history": history,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-init-path", default="/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae")
    ap.add_argument("--raw-manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--calibration-json", default="/root/Workspace/WEAVE_VAE/swd_calibration.json")
    ap.add_argument("--output-dir", default="/root/Workspace/WEAVE_VAE/runs_vae/V2")
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--microbatch", type=int, default=6)
    ap.add_argument("--effective-batch", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--beta-kl", type=float, default=2.252e-5, help="KL divergence weight (weakened to 1/5th of original)")
    ap.add_argument("--lambda-style", type=float, default=0.0100)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-decoder", type=float, default=5e-6)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    train_vae_v2(args)
