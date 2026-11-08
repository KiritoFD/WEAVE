"""WEAVE V7 P2-fix: corrected DMD sign + smooth residual + unified output.

Trains a 1-NFE SingleStepTransportNet with fixed S1 real ScoreNet.
Two same-budget variants are controlled by --lambda-content:
  - pure_dmd:   lambda_content = 0
  - dmd_content: lambda_content > 0

Key fixes versus d5100c0:
  1. Corrected DMD target sign:
         direction = (eps_real - eps_fake) / sigma
         target = y.detach() - alpha * direction
     Gradient descent on 0.5*MSE(y,target) moves q_theta toward p_real.
  2. Smooth, differentiable residual bound:
         T(z)=z+c*tanh(R_theta(z)/c)
     implemented inside SingleStepTransportNet; training and inference both call
     transport(...) directly, with no Straight-Through clamp.
  3. Removed invalid same-style identity loss (s==t is exact by mask).
  4. Logs separate generator gradient norms for DMD and content losses.
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

import paths  # noqa: E402
from tools.v6.models_v6 import ConditionalScoreNet  # noqa: E402
from tools.v7.common import STYLES, V7_ROOT, ensure_v7_dirs, save_json  # noqa: E402
from tools.v7.models_v7 import SingleStepTransportNet  # noqa: E402
from tools.v7.p2_train_rdmd import content_transport_cost  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v7.common import V6_ROOT  # noqa: E402


def grad_global_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().norm(p=2).item() ** 2)
    return float(sq ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lambda-content", type=float, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--dmd-alpha", type=float, default=0.10)
    ap.add_argument("--residual-cap", type=float, default=0.35)
    ap.add_argument("--sigma-dmd", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_v7_dirs()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V7 P2-fix [{args.name}] Device={device} "
          f"lambda_content={args.lambda_content} alpha={args.dmd_alpha} ===", flush=True)

    asset_root = paths.asset_root()
    orig_dir = asset_root / "pseudo_gt_24h" / "orig_mean_fp16"
    gpu_by_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(orig_dir / f"{sid:02d}_{s}.pt", map_location="cpu", weights_only=False)
        gpu_by_style[s] = d["latents"].float()[100:].to(device)

    # Frozen real Score = S1 pure denoising MSE estimator.
    real_score = ConditionalScoreNet(base_dim=64, num_styles=len(STYLES)).to(device).eval()
    real_ck = torch.load(V7_ROOT / "scorenet_s1_pure_mse.pt", map_location="cpu", weights_only=False)
    real_score.load_state_dict(real_ck["model"])
    for p in real_score.parameters():
        p.requires_grad_(False)

    # Fake Score starts exactly at S1 and adapts online to the generator distribution.
    fake_score = ConditionalScoreNet(base_dim=64, num_styles=len(STYLES)).to(device)
    fake_score.load_state_dict(real_ck["model"])

    transport = SingleStepTransportNet(
        base_dim=48, num_styles=len(STYLES), residual_cap=args.residual_cap
    ).to(device)
    canon = ContentCanonicalizer(
        torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    ).to(device).eval()

    opt_g = torch.optim.AdamW(transport.parameters(), lr=2e-4, weight_decay=1e-4)
    opt_f = torch.optim.AdamW(fake_score.parameters(), lr=1e-4, weight_decay=1e-4)

    per_style = 8
    B = per_style * len(STYLES)
    sig_vec = torch.full((B,), args.sigma_dmd, device=device, dtype=torch.float32)

    hist: Dict[str, List[float]] = {
        "fake": [], "dmd": [], "content": [], "total": [],
        "grad_dmd": [], "grad_content": [], "grad_total": [],
        "residual_l2": [],
    }
    t0 = time.time()

    for step in range(1, args.steps + 1):
        z_list, src_list = [], []
        for sid, s in enumerate(STYLES):
            idx = torch.randint(0, gpu_by_style[s].shape[0], (per_style,), device=device)
            z_list.append(gpu_by_style[s][idx])
            src_list.append(torch.full((per_style,), sid, device=device, dtype=torch.long))
        z = torch.cat(z_list, dim=0)
        src_ids = torch.cat(src_list, dim=0)
        shift = torch.randint(1, len(STYLES), (B,), device=device)
        tgt_ids = (src_ids + shift) % len(STYLES)

        # Unified training/inference output function: no clamp wrapper.
        y = transport(z, src_ids, tgt_ids)

        # 1) Fake ScoreNet update on detached generator outputs.
        eps_f = torch.randn_like(y)
        y_sig_f = y.detach() + args.sigma_dmd * eps_f
        eps_fake_pred = fake_score(y_sig_f, sig_vec, tgt_ids)
        loss_fake = F.mse_loss(eps_fake_pred.float(), eps_f)
        opt_f.zero_grad(set_to_none=True)
        loss_fake.backward()
        torch.nn.utils.clip_grad_norm_(fake_score.parameters(), 1.0)
        opt_f.step()

        # 2) Generator update with corrected DMD direction.
        y = transport(z, src_ids, tgt_ids)
        with torch.no_grad():
            eps_d = torch.randn_like(y)
            y_sig = y.detach() + args.sigma_dmd * eps_d
            eps_real = real_score(y_sig, sig_vec, tgt_ids)
            eps_fake = fake_score(y_sig, sig_vec, tgt_ids)
            direction = (eps_real - eps_fake) / args.sigma_dmd
            target = y.detach() - args.dmd_alpha * direction
        loss_dmd = 0.5 * F.mse_loss(y.float(), target.float())

        loss_content = content_transport_cost(z, y, canon) if args.lambda_content > 0 else y.new_tensor(0.0)
        loss_total = loss_dmd + args.lambda_content * loss_content

        # Separate gradient norms (actual generator parameter gradients).
        opt_g.zero_grad(set_to_none=True)
        loss_dmd.backward(retain_graph=(args.lambda_content > 0))
        g_dmd = grad_global_norm(transport.parameters())
        if args.lambda_content > 0:
            opt_g.zero_grad(set_to_none=True)
            (args.lambda_content * loss_content).backward(retain_graph=True)
            g_content = grad_global_norm(transport.parameters())

            opt_g.zero_grad(set_to_none=True)
            loss_total.backward()
        else:
            g_content = 0.0
            # loss_dmd.backward already populated the final gradient.
        g_total = grad_global_norm(transport.parameters())
        torch.nn.utils.clip_grad_norm_(transport.parameters(), 1.0)
        opt_g.step()

        hist["fake"].append(float(loss_fake.item()))
        hist["dmd"].append(float(loss_dmd.item()))
        hist["content"].append(float(loss_content.item()))
        hist["total"].append(float(loss_total.item()))
        hist["grad_dmd"].append(g_dmd)
        hist["grad_content"].append(g_content)
        hist["grad_total"].append(g_total)
        hist["residual_l2"].append(float((y.detach() - z).flatten(1).norm(p=2, dim=1).mean().item()))

        if step == 1 or step % 100 == 0 or step == args.steps:
            peak = torch.cuda.max_memory_allocated(0) / (1024**2)
            print(f"[{args.name}] step {step:4d}/{args.steps} fake={np.mean(hist['fake'][-50:]):.4f} "
                  f"dmd={np.mean(hist['dmd'][-50:]):.4f} content={np.mean(hist['content'][-50:]):.4f} "
                  f"|g_dmd|={np.mean(hist['grad_dmd'][-50:]):.3f} |g_content|={np.mean(hist['grad_content'][-50:]):.3f} "
                  f"|g_total|={np.mean(hist['grad_total'][-50:]):.3f} resL2={np.mean(hist['residual_l2'][-50:]):.3f} "
                  f"peak={peak:.0f}MiB elapsed={time.time()-t0:.1f}s", flush=True)

    out_ckpt = V7_ROOT / f"{args.name}.pt"
    torch.save({
        "transport": transport.state_dict(),
        "fake_score": fake_score.state_dict(),
        "config": {
            "name": args.name,
            "base_dim": 48,
            "steps": args.steps,
            "batch": B,
            "dmd_alpha": args.dmd_alpha,
            "lambda_content": args.lambda_content,
            "residual_cap": args.residual_cap,
            "sigma_dmd": args.sigma_dmd,
            "real_score": "S1 pure denoising MSE frozen",
            "output": "z + mask*c*tanh(R/c), identical train/inference",
            "dmd_sign": "target = y - alpha*(eps_real-eps_fake)/sigma",
        },
    }, out_ckpt)

    save_json({
        "name": args.name,
        "lambda_content": args.lambda_content,
        "steps": args.steps,
        "batch": B,
        "dmd_alpha": args.dmd_alpha,
        "residual_cap": args.residual_cap,
        "sigma_dmd": args.sigma_dmd,
        "train_wall_sec": round(time.time() - t0, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "final": {k: round(float(np.mean(v[-50:])), 5) for k, v in hist.items()},
        "checkpoint": str(out_ckpt.relative_to(_REPO_ROOT)),
    }, V7_ROOT / f"{args.name}_train_summary.json")
    print(f"[{args.name}] Saved {out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
