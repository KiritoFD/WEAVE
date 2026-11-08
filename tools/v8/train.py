"""V8 B0-B3 unified trainer: random/OT endpoint coupling x CFM/Brownian bridge.

No ScoreNet, no DMD, no GAN, no RGB reconstruction. Inference uses Euler/Heun
solvers with <=16 NFE.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v8.bridge import sample_bridge  # noqa: E402
from tools.v8.common import REPO_ROOT as V8_REPO_ROOT, STYLES, V8_ROOT, ensure_dirs, load_d5_latents, save_json  # noqa: E402
from tools.v8.models import VelocityUNet  # noqa: E402
from tools.v8.pair_sampler import ot_sample_targets  # noqa: E402

V6_ROOT = V8_REPO_ROOT / "runs" / "v6"


def ema_update(ema: Dict[str, torch.Tensor], model: torch.nn.Module, decay: float = 0.999) -> None:
    with torch.no_grad():
        for k, p in model.state_dict().items():
            if not torch.is_tensor(p) or not p.is_floating_point():
                continue
            if k not in ema:
                ema[k] = p.detach().clone()
            else:
                ema[k].mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def sample_batch(gpu_by_style, per_style, device, gen: torch.Generator):
    """Balanced source batch from a step-wise deterministic generator."""
    z0_list = []
    for s in STYLES:
        idx = torch.randint(0, gpu_by_style[s].shape[0], (per_style,), generator=gen, device=device)
        z0_list.append(gpu_by_style[s][idx])
    return torch.cat(z0_list, dim=0)


def make_generator(device: str, base_seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(base_seed))
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=["B0", "B1", "B2", "B3"])
    ap.add_argument("--tag", default="", help="Optional output subdirectory for short sweeps")
    ap.add_argument("--pairing", choices=["random", "ot"], required=True)
    ap.add_argument("--sigma-mid", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8, help="effective batch=microbatch*accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ot-reg", type=float, default=0.08)
    args = ap.parse_args()

    ensure_dirs()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== V8 {args.name} | pairing={args.pairing} sigma_mid={args.sigma_mid} Device={device} ===", flush=True)

    gpu_by_style, val_by_style, data_meta = load_d5_latents(device=device, heldout=100)
    model = VelocityUNet(base_dim=48, num_styles=len(STYLES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    canon = None
    if args.pairing == "ot":
        stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
        canon = ContentCanonicalizer(stats).to(device).eval()

    B = args.microbatch
    per_style = max(1, B // len(STYLES))
    B = per_style * len(STYLES)
    src_ids = torch.cat([torch.full((per_style,), sid, device=device, dtype=torch.long) for sid in range(len(STYLES))], 0)
    ema: Dict[str, torch.Tensor] = {}
    # Per-step generators are seeded by (global seed, step), so B0-B3 consume the
    # same source/target/tau/noise sequence for an identical update count N.
    hist: Dict[str, List[float]] = {"loss": [], "target_rms": [], "grad": [], "time_low": [], "time_mid": [], "time_high": []}
    ot_hist: List[dict] = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        step_gen = make_generator(device, args.seed * 1000003 + step)
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        step_target_rms = []
        time_buckets = {"low": [], "mid": [], "high": []}
        for a in range(args.accum):
            z0 = sample_batch(gpu_by_style, per_style, device, step_gen)
            shift = torch.randint(1, len(STYLES), (B,), generator=step_gen, device=device)
            tgt_ids = (src_ids + shift) % len(STYLES)

            # Sample target endpoints from corresponding target pools.
            z1 = torch.empty_like(z0)
            for tid in range(len(STYLES)):
                mask = tgt_ids == tid
                if not mask.any():
                    continue
                pool = gpu_by_style[STYLES[tid]]
                idx = torch.randint(0, pool.shape[0], (int(mask.sum().item()),), generator=step_gen, device=device)
                z1[mask] = pool[idx]

            ot_info = None
            if args.pairing == "ot":
                # OT is applied per target-style subgroup; all endpoints remain real.
                z1_ot = torch.empty_like(z1)
                for tid in range(len(STYLES)):
                    mask = tgt_ids == tid
                    if not mask.any():
                        continue
                    pool = gpu_by_style[STYLES[tid]]
                    ot_batch = min(128, pool.shape[0])
                    oidx = torch.randint(0, pool.shape[0], (ot_batch,), generator=step_gen, device=device)
                    matched, info = ot_sample_targets(
                        z0[mask], pool[oidx], canon, reg=args.ot_reg,
                        seed=int(torch.randint(0, 2**31 - 1, (1,), generator=step_gen, device=device).item()),
                    )
                    z1_ot[mask] = matched
                    ot_info = info
                z1 = z1_ot

            tau = 0.01 + 0.98 * torch.rand(B, generator=step_gen, device=device)
            bridge_noise = torch.randn(z0.shape, generator=step_gen, device=device, dtype=z0.dtype)
            z_tau, v_target = sample_bridge(z0, z1, tau, sigma_mid=args.sigma_mid, noise=bridge_noise)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                v_pred = model(z_tau, tau, src_ids, tgt_ids)
                loss = F.mse_loss(v_pred.float(), v_target.float()) / args.accum
            loss.backward()
            total_loss += float(loss.item())
            step_target_rms.append(float(v_target.float().pow(2).mean().sqrt().item()))

            err = (v_pred.detach().float() - v_target.float()).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
            tn = tau.detach().cpu().numpy()
            time_buckets["low"].extend(err[tn < 0.33].tolist())
            time_buckets["mid"].extend(err[(tn >= 0.33) & (tn < 0.67)].tolist())
            time_buckets["high"].extend(err[tn >= 0.67].tolist())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        ema_update(ema, model, 0.999)

        hist["loss"].append(total_loss)
        hist["target_rms"].append(float(np.mean(step_target_rms)))
        hist["grad"].append(float(grad_norm.item()))
        for k in ("low", "mid", "high"):
            hist[f"time_{k}"].append(float(np.mean(time_buckets[k])) if time_buckets[k] else 0.0)
        if ot_info is not None and step % 100 == 0:
            ot_hist.append({"step": step, **ot_info})

        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"[V8 {args.name}] step {step:4d}/{args.steps} loss={np.mean(hist['loss'][-50:]):.4f} "
                  f"target_rms={np.mean(hist['target_rms'][-50:]):.3f} grad={np.mean(hist['grad'][-50:]):.3f} "
                  f"mse[low/mid/high]={np.mean(hist['time_low'][-50:]):.3f}/{np.mean(hist['time_mid'][-50:]):.3f}/{np.mean(hist['time_high'][-50:]):.3f} "
                  f"peak={torch.cuda.max_memory_allocated(0)/(1024**2):.0f}MiB elapsed={time.time()-t0:.1f}s", flush=True)

    out = V8_ROOT / (args.tag or args.name)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema,
        "config": vars(args) | {"n_params": n_params, "effective_batch": B * args.accum, "train_wall_sec": round(time.time() - t0, 1)},
    }, out / "model.pt")
    save_json({
        "name": args.name,
        "config": vars(args) | {"n_params": n_params, "effective_batch": B * args.accum},
        "data_meta": data_meta,
        "train_wall_sec": round(time.time() - t0, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "final": {k: round(float(np.mean(v[-50:])), 5) for k, v in hist.items()},
        "ot_checks": ot_hist,
    }, out / "train_summary.json")
    print(f"[V8 {args.name}] Saved {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
