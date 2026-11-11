"""WEAVE W P1 — train an unconditional real-latent score/denoiser prior.

Trains a ConditionalScoreNet (with a dummy single style) on the A0-encoded
15,545-image training pool to approximate the score of the real latent
distribution:

    z_sigma = z + sigma * eps
    L = ||eps_theta(z_sigma, sigma) - eps||^2
    score(z, sigma) ~= -eps_theta(z, sigma) / sigma

This is the "real data score" term for the next guided-editing experiment. The
existing 5-style ScoreDiff uses margin losses and style conditioning; this model
is deliberately unconditional so it can be combined with R1's target-style
gradient without double-counting style supervision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v6.models_v6 import ConditionalScoreNet  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def random_crop_batch(z: torch.Tensor, crop_size: int) -> torch.Tensor:
    if crop_size >= z.shape[-1]:
        return z
    b, c, h, w = z.shape
    top = torch.randint(0, h - crop_size + 1, (1,), device=z.device).item()
    left = torch.randint(0, w - crop_size + 1, (1,), device=z.device).item()
    return z[:, :, top : top + crop_size, left : left + crop_size]


def main():
    ap = argparse.ArgumentParser(description="Train unconditional latent score prior")
    ap.add_argument("--latent-dir", default="pseudo_gt_24h/orig_mean_fp16_a0")
    ap.add_argument("--out-dir", default="runs/v6_w/score_prior")
    ap.add_argument("--base-dim", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--microbatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_dir = ROOT / args.latent_dir
    files = sorted(latent_dir.glob("[0-9][0-9]_*.pt"))
    if not files:
        raise FileNotFoundError(f"No A0 latent pool files under {latent_dir}")
    print(f"[ScorePrior] loading {len(files)} style files from {latent_dir}", flush=True)
    chunks = []
    n_total = 0
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        x = d["latents"].float()
        chunks.append(x)
        n_total += x.shape[0]
        print(f"  loaded {s if (s:=d.get('subdir')) else p.name}: {x.shape[0]}", flush=True)
    z_all = torch.cat(chunks, dim=0)
    del chunks
    print(f"[ScorePrior] total latents: {n_total}", flush=True)

    # Hold out a small deterministic validation subset for a denoising sanity check.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(z_all.shape[0], generator=g)
    val_idx = perm[:1000]
    train_idx = perm[1000:]
    z_train = z_all[train_idx].to(device)
    z_val = z_all[val_idx].to(device)

    model = ConditionalScoreNet(base_dim=args.base_dim, num_styles=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    dummy = torch.zeros(1, dtype=torch.long, device=device)

    sigmas = [0.15, 0.30, 0.50, 0.80]
    sig_t = torch.tensor(sigmas, device=device, dtype=torch.float32)
    crops = [16, 32, 64]
    loss_hist = []
    t0 = time.time()
    print(f"[ScorePrior] params={n_params/1e6:.2f}M steps={args.steps} "
          f"microbatch={args.microbatch}", flush=True)

    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, z_train.shape[0], (args.microbatch,), device=device)
        z = z_train[idx]
        crop = crops[(step - 1) % len(crops)]
        z = random_crop_batch(z, crop)
        sigma = sig_t[torch.randint(0, len(sigmas), (z.shape[0],), device=device)]
        eps = torch.randn_like(z)
        z_noisy = z + sigma.view(-1, 1, 1, 1) * eps
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            eps_pred = model(z_noisy, sigma, dummy.expand(z.shape[0]))
            loss = F.mse_loss(eps_pred.float(), eps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_hist.append(float(loss.detach()))
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"[ScorePrior] step {step}/{args.steps} "
                  f"loss={np.mean(loss_hist[-100:]):.4f} "
                  f"peak_vram={torch.cuda.max_memory_allocated(0)/1024**2:.0f}MiB "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                enabled=(device == "cuda")):
        val_losses = []
        for sigma in sigmas:
            eps = torch.randn_like(z_val)
            z_noisy = z_val + sigma * eps
            pred = model(z_noisy, torch.full((z_val.shape[0],), sigma, device=device),
                         torch.zeros(z_val.shape[0], dtype=torch.long, device=device))
            val_losses.append(float(F.mse_loss(pred.float(), eps)))
    stats = {
        "n_params": int(n_params),
        "steps": args.steps,
        "train_wall_sec": round(time.time() - t0, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
        "final_train_loss": float(np.mean(loss_hist[-50:])),
        "val_denoise_mse_by_sigma": dict(zip(map(str, sigmas), val_losses)),
        "n_train": int(z_train.shape[0]),
        "n_val": int(z_val.shape[0]),
        "latent_dir": str(latent_dir),
    }
    ckpt = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": vars(args),
        "stats": stats,
    }
    torch.save(ckpt, out_dir / "score_prior.pt")
    (out_dir / "train_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[ScorePrior] saved {out_dir/'score_prior.pt'}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
