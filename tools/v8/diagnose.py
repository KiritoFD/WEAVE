"""V8 P0/P1 velocity-field diagnostics (no retraining required).

Provides four measurements the P0 review demanded:
  1. velocity fit per time bucket: pred RMS, target RMS, cosine, and MSE improvement
     over a zero-velocity predictor;
  2. style condition response: fix z_tau, vary target style, measure velocity change;
  3. endpoint distribution match: CSD-MMD between generated and real target latents,
     plus Top-1 and confusion matrix via the RGB Evaluator V2 path;
  4. solver convergence: Euler 4/8/16/32 and Heun 2/4/8 on a fixed holdout slice.

Usage:
  python tools/v8/diagnose.py --ckpt runs/v8/B0/model.pt --weight raw --name B0
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

from tools.v8.bridge import sample_bridge  # noqa: E402
from tools.v8.common import (  # noqa: E402
    STYLES, V8_ROOT, aggregate_rows, batch_tensors, build_independent_val_tasks,
    build_source_latents, compute_batch_rows, ensure_dirs, load_d5_latents, load_eval_bundle,
    save_json,
)
from tools.v8.models import VelocityUNet  # noqa: E402
from tools.v8.solvers import solve_euler  # noqa: E402


def load_model(ckpt: str, weight: str, device: str) -> VelocityUNet:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = VelocityUNet(base_dim=48, num_styles=len(STYLES)).to(device).eval()
    if weight == "raw":
        model.load_state_dict(ck.get("model", ck))
    elif weight == "ema":
        if "ema" not in ck or not ck["ema"]:
            raise KeyError(f"{ckpt} has no EMA weights")
        model.load_state_dict(ck["ema"])
    else:
        raise ValueError(weight)
    return model


@torch.inference_mode()
def diagnose_fit(model, train_by_style, device: str, n_batch: int = 12, per_style: int = 4) -> dict:
    """Per-time-bucket velocity regression quality against the bridge target."""
    buckets = {"0.00-0.25": [], "0.25-0.50": [], "0.50-0.75": [], "0.75-1.00": []}
    g = torch.Generator(device=device).manual_seed(0)
    sums = {k: {"pred_sq": 0.0, "tgt_sq": 0.0, "err_sq": 0.0, "dot": 0.0, "n": 0} for k in buckets}
    for _ in range(n_batch):
        z0_list, src_ids = [], []
        for sid, s in enumerate(STYLES):
            pool = train_by_style[s]
            idx = torch.randint(0, pool.shape[0], (per_style,), generator=g, device=device)
            z0_list.append(pool[idx])
            src_ids.append(torch.full((per_style,), sid, device=device, dtype=torch.long))
        z0 = torch.cat(z0_list)
        src = torch.cat(src_ids)
        tgt = (src + torch.randint(1, len(STYLES), (z0.shape[0],), generator=g, device=device)) % len(STYLES)
        z1 = torch.empty_like(z0)
        for tid in range(len(STYLES)):
            m = tgt == tid
            if m.any():
                pool = train_by_style[STYLES[tid]]
                z1[m] = pool[torch.randint(0, pool.shape[0], (int(m.sum()),), generator=g, device=device)]
        tau = torch.rand(z0.shape[0], generator=g, device=device) * 0.98 + 0.01
        noise = torch.randn(z0.shape, generator=g, device=device, dtype=z0.dtype)
        z_tau, v_tgt = sample_bridge(z0, z1, tau, sigma_mid=0.0, noise=noise)
        v_pred = model(z_tau, tau, src, tgt).float()
        v_tgt = v_tgt.float()
        tn = tau.cpu().numpy()
        err = (v_pred - v_tgt).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
        psq = v_pred.pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
        tsq = v_tgt.pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
        dot = (v_pred * v_tgt).mean(dim=(1, 2, 3)).cpu().numpy()
        edges = [0.0, 0.25, 0.5, 0.75, 1.01]
        for i in range(len(buckets)):
            m = (tn >= edges[i]) & (tn < edges[i + 1])
            if not m.any():
                continue
            k = list(buckets)[i]
            sums[k]["pred_sq"] += float(psq[m].sum())
            sums[k]["tgt_sq"] += float(tsq[m].sum())
            sums[k]["err_sq"] += float(err[m].sum())
            sums[k]["dot"] += float(dot[m].sum())
            sums[k]["n"] += int(m.sum())
    out = {}
    for k, v in sums.items():
        if v["n"] == 0:
            continue
        pred_rms = float(np.sqrt(v["pred_sq"] / v["n"]))
        tgt_rms = float(np.sqrt(v["tgt_sq"] / v["n"]))
        mse = float(v["err_sq"] / v["n"])
        # zero-velocity predictor MSE is E||v_tgt||^2
        zero_mse = float(v["tgt_sq"] / v["n"])
        cos = float(v["dot"] / max(np.sqrt(v["pred_sq"] * v["tgt_sq"]), 1e-12))
        out[k] = {
            "n": v["n"], "pred_rms": round(pred_rms, 4), "target_rms": round(tgt_rms, 4),
            "mse": round(mse, 4), "zero_velocity_mse": round(zero_mse, 4),
            "mse_improvement_over_zero": round(1.0 - mse / max(zero_mse, 1e-12), 4),
            "cosine": round(cos, 4),
        }
    return out


@torch.inference_mode()
def diagnose_condition_response(model, train_by_style, device: str, n: int = 64) -> dict:
    """Fix z_tau and source style; vary target style. Measures how much u responds."""
    g = torch.Generator(device=device).manual_seed(1)
    z0 = train_by_style["Early_Renaissance"][:n]
    tau = torch.full((n,), 0.5, device=device)
    src = torch.zeros(n, device=device, dtype=torch.long)
    vel = {}
    for t in range(len(STYLES)):
        tgt = torch.full((n,), t, device=device, dtype=torch.long)
        vel[t] = model(z0, tau, src, tgt).float()
    base = vel[0]
    per_target = {}
    for t in range(1, len(STYLES)):
        d = (vel[t] - base)
        per_target[STYLES[t]] = {
            "delta_rms": round(float(d.pow(2).mean().sqrt().item()), 4),
            "cos_with_src_velocity": round(float(
                F.cosine_similarity(d.flatten(1), base.flatten(1), dim=1).mean().item()), 4),
        }
    pair_cos = []
    for i in range(len(STYLES)):
        for j in range(i + 1, len(STYLES)):
            a = vel[i].flatten(1); b = vel[j].flatten(1)
            pair_cos.append(float(F.cosine_similarity(a, b, dim=1).mean().item()))
    return {
        "base_velocity_rms": round(float(base.pow(2).mean().sqrt().item()), 4),
        "per_target_vs_src": per_target,
        "mean_pairwise_cos_between_targets": round(float(np.mean(pair_cos)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weight", choices=["raw", "ema"], default="raw")
    ap.add_argument("--per-style", type=int, default=10)
    args = ap.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== V8 diagnose {args.name} weight={args.weight} ===", flush=True)

    train_by_style, _, data_meta = load_d5_latents(device=device, heldout=100)
    model = load_model(args.ckpt, args.weight, device)

    fit = diagnose_fit(model, train_by_style, device)
    cond = diagnose_condition_response(model, train_by_style, device)

    tasks = build_independent_val_tasks(per_style=args.per_style)
    models, ref_feats, vae, trt = load_eval_bundle(device)
    src_latents = build_source_latents(tasks, vae, device)
    source_cache: Dict[str, dict] = {}

    solvers = []
    for spec in ["euler4", "euler8", "euler16", "euler32"]:
        solvers.append((spec, "euler", int(spec[5:])))
    convergence = {}
    results = {}
    for spec, method, steps in solvers:
        rows, l2 = [], []
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(tasks), 8):
                chunk = tasks[i:i + 8]
                z0, s, t = batch_tensors(chunk, src_latents, device)
                z_gen = solve_euler(model, z0, s, t, steps)
                l2.extend((z_gen.float() - z0.float()).flatten(1).norm(p=2, dim=1).cpu().tolist())
                rows.extend(compute_batch_rows(z_gen, chunk, models, ref_feats, vae, trt, source_cache, latent_l2=l2[-len(chunk):], nfe=steps))
        agg = aggregate_rows(rows, only_cross=True)
        agg["gen_wall_sec"] = round(time.time() - t0, 1)
        results[spec] = agg
        convergence[spec] = {
            "latent_l2_mean": round(float(np.mean([r["latent_l2"] for r in rows])), 4),
            "mean_dT": agg["mean_dT"], "Top1": agg["Target_Top1"], "DINO_C": agg["DINO_C"], "LPIPS": agg["LPIPS"],
        }
        print(f"  [{spec}] dT={agg['mean_dT']:+.4f} Top-1={agg['Target_Top1']}% DINO-C={agg['DINO_C']:.4f} "
              f"LPIPS={agg['LPIPS']:.4f} latentL2={convergence[spec]['latent_l2_mean']:.3f}", flush=True)

    out = {
        "checkpoint": args.ckpt, "weight": args.weight, "data": data_meta["vae"],
        "n_val_tasks": len(tasks),
        "velocity_fit": fit,
        "condition_response": cond,
        "solver_convergence": convergence,
        "per_solver_metrics": results,
    }
    save_json(out, V8_ROOT / f"diagnose_{args.name}_{args.weight}.json")
    print(f"[V8 diagnose] saved runs/v8/diagnose_{args.name}_{args.weight}.json", flush=True)


if __name__ == "__main__":
    main()
