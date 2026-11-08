"""WEAVE V7 P0: Tweedie one-step / few-step difference scan.

Question:
    Are 12-24 Score Difference integration steps actually necessary?
    We compare:
      - Calib_Balanced_s16 baseline (16 steps x 3 sigmas x 2 ScoreNet calls = 96 NFE)
      - Tweedie posterior-mean difference:
            T(z,s,t) = z + alpha * E_eps [ D_t(z+sigma*eps,sigma) - D_s(z+sigma*eps,sigma) ]
                      = z + alpha * sigma * E_eps [ eps_s - eps_t ]
        with K = 1 or 4 deterministic noise samples (2 or 8 ScoreNet NFE).

Evaluation:
    Strict independent validation board built from distinct5_512/train, excluding
    all frozen test source stems. No Test600 tuning is performed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution: `python3 /path/WEAVE/tools/v7/p0_tweedie_scan.py`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from tools.v7.common import (
    DOCS_V7,
    STYLES,
    V7_ROOT,
    aggregate_rows,
    batch_tensors,
    build_independent_val_tasks,
    build_source_latents,
    compute_batch_rows,
    ensure_v7_dirs,
    load_eval_bundle,
    load_score_model,
    save_json,
    task_seed,
)
from tools.v6.score_diff import edit_with_score_difference

CALIB_SCALES = {
    "Minimalism": 0.50,
    "Ukiyo_e": 1.60,
    "Rococo": 1.40,
    "Impressionism": 1.30,
    "Early_Renaissance": 1.20,
}
SIGMAS = [0.30, 0.50, 0.80]
NOISE_COUNTS = [1, 4]
ALPHA_MULTIPLIERS = [0.5, 1.0, 2.0]
BATCH_SIZE = 16


def run_generator(
    gen_fn,
    tasks: Sequence[dict],
    src_latents: Dict[str, torch.Tensor],
    models,
    ref_feats,
    vae,
    trt,
    source_cache: Dict[str, dict],
    device: str,
) -> tuple[List[dict], float, List[float]]:
    rows: List[dict] = []
    l2_all: List[float] = []
    t0 = time.time()
    for i in range(0, len(tasks), BATCH_SIZE):
        chunk = tasks[i : i + BATCH_SIZE]
        z, s, t, task_ids = batch_tensors(chunk, src_latents, device)
        z_gen = gen_fn(z, s, t, task_ids)
        l2 = (z_gen.float() - z.float()).flatten(1).norm(p=2, dim=1).cpu().tolist()
        l2_all.extend(l2)
        rows.extend(compute_batch_rows(z_gen, chunk, models, ref_feats, vae, trt, source_cache, latent_l2=l2))
    wall = time.time() - t0
    return rows, wall, l2_all


@torch.inference_mode()
def tweedie_raw_diff(
    model,
    z: torch.Tensor,
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    task_ids: Sequence[str],
    sigma: float,
    n_noise: int,
    device: str,
) -> torch.Tensor:
    """Return sigma * mean_k (eps_theta(y_sigma,s) - eps_theta(y_sigma,t))."""
    B = z.shape[0]
    acc = torch.zeros_like(z)
    for k in range(n_noise):
        eps = torch.stack([
            torch.randn(
                z[j].shape,
                generator=torch.Generator(device=device).manual_seed(task_seed(task_ids[j], k)),
                device=device,
                dtype=z.dtype,
            )
            for j in range(B)
        ], dim=0)
        y_sigma = z + sigma * eps
        eps_s = model(y_sigma, sigma, src_ids)
        eps_t = model(y_sigma, sigma, tgt_ids)
        acc += sigma * (eps_s - eps_t) / float(n_noise)
    return acc


def main() -> None:
    ensure_v7_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V7 P0 Tweedie One-Step Scan (Device: {device}) ===", flush=True)

    tasks = build_independent_val_tasks(per_style=10)
    print(f"[V7 P0] Independent validation tasks: {len(tasks)} "
          f"({len(set(t['source_stem'] for t in tasks))} train-split sources, fully test-disjoint)", flush=True)

    models, ref_feats, vae, trt = load_eval_bundle(device)
    sd_model = load_score_model(device=device)
    src_latents = build_source_latents(tasks, vae, device)
    source_cache: Dict[str, dict] = {}

    results: dict = {
        "protocol": {
            "validation": "independent distinct5_512/train sources, disjoint from frozen Test600 sources",
            "n_tasks": len(tasks),
            "n_sources": len(set(t["source_stem"] for t in tasks)),
            "noise": "deterministic task_id + noise_idx hash seed",
            "ref_pool": "eval_protocol/references_v1.json",
            "baseline": "Calib_Balanced_s16 (16 steps, step_size=0.120, anchor=0.08, style-specific scales)",
        },
        "baselines": {},
        "tweedie": {},
        "best": {},
    }

    # 1. VAE No-op reference (quantifies VAE inherent distortion under paper_rgb_v1)
    print("\n--- Evaluating VAE No-op (zero editor drift) ---", flush=True)
    noop_rows, noop_wall, noop_l2 = run_generator(
        lambda z, s, t, ids: z,
        tasks, src_latents, models, ref_feats, vae, trt, source_cache, device,
    )
    noop_agg = aggregate_rows(noop_rows)
    noop_agg.update({"NFE": 0, "gen_wall_sec": round(noop_wall, 1), "mean_latent_l2": 0.0})
    results["baselines"]["VAE_Noop"] = noop_agg
    print(f"  [VAE No-op] dT={noop_agg['mean_dT']:+.4f} Top-1={noop_agg['Target_Top1']}% "
          f"DINO-C={noop_agg['DINO_C']:.4f} LPIPS={noop_agg['LPIPS']:.4f} ukiyo_dT={noop_agg['ukiyo_dT']:+.4f}", flush=True)

    # 2. Calib_Balanced_s16 baseline on the SAME independent validation tasks
    print("\n--- Evaluating Calib_Balanced_s16 (96 NFE) ---", flush=True)
    def calib_gen(z, s, t, ids):
        return edit_with_score_difference(
            sd_model, z, s, t,
            num_steps=16, step_size=0.120, struct_anchor_weight=0.08,
            seed=42, task_ids=ids, style_step_scales=CALIB_SCALES,
        )

    base_rows, base_wall, base_l2 = run_generator(
        calib_gen, tasks, src_latents, models, ref_feats, vae, trt, source_cache, device,
    )
    base_agg = aggregate_rows(base_rows)
    base_agg.update({"NFE": 96, "gen_wall_sec": round(base_wall, 1), "mean_latent_l2": round(float(np.mean(base_l2)), 4)})
    results["baselines"]["Calib_Balanced_s16"] = base_agg
    target_l2 = float(np.mean(base_l2))
    print(f"  [Calib_Balanced_s16] dT={base_agg['mean_dT']:+.4f} Top-1={base_agg['Target_Top1']}% "
          f"Margin={base_agg['Target_Margin']:+.4f} DINO-C={base_agg['DINO_C']:.4f} "
          f"LPIPS={base_agg['LPIPS']:.4f} ukiyo_Top1={base_agg['ukiyo_top1']}% "
          f"mean_latent_l2={target_l2:.4f} gen={base_wall:.1f}s", flush=True)

    # 3. First pass: compute deterministic raw Tweedie differences and their L2 norm
    print("\n--- Precomputing Tweedie raw displacement fields ---", flush=True)
    raw_cache: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    raw_norm: Dict[str, float] = {}
    for sigma in SIGMAS:
        for n_noise in NOISE_COUNTS:
            key = f"sigma{sigma:.2f}_K{n_noise}"
            diffs: List[torch.Tensor] = []
            norms: List[float] = []
            t0 = time.time()
            for i in range(0, len(tasks), BATCH_SIZE):
                chunk = tasks[i : i + BATCH_SIZE]
                z, s, t, ids = batch_tensors(chunk, src_latents, device)
                diff = tweedie_raw_diff(sd_model, z, s, t, ids, sigma, n_noise, device)
                norms.extend(diff.flatten(1).norm(p=2, dim=1).cpu().tolist())
                diffs.extend(list(diff.cpu()))
            raw_cache[key] = {"diffs": diffs}
            raw_norm[key] = float(np.mean(norms))
            print(f"  [{key}] raw_L2={raw_norm[key]:.4f} NFE={2*n_noise} "
                  f"precompute={time.time()-t0:.1f}s alpha0={target_l2/max(raw_norm[key],1e-6):.3f}", flush=True)

    # 4. Second pass: alpha scan by matching baseline latent displacement norm
    print("\n--- Scanning calibrated alpha multipliers ---", flush=True)
    noop_ukiyo = float(noop_agg["per_target_dT"]["Ukiyo_e"])
    for sigma in SIGMAS:
        for n_noise in NOISE_COUNTS:
            key = f"sigma{sigma:.2f}_K{n_noise}"
            alpha0 = target_l2 / max(raw_norm[key], 1e-6)
            diffs = raw_cache[key]["diffs"]

            for mult in ALPHA_MULTIPLIERS:
                alpha = alpha0 * mult
                cfg = f"{key}_alpha{alpha:.3f}_x{mult:.1f}"

                def make_gen(alpha_v=alpha, diff_list=diffs, n=n_noise):
                    def _gen(z, s, t, ids, offset: List[int] = [0]):
                        b = z.shape[0]
                        d = torch.stack(diff_list[offset[0] : offset[0] + b], dim=0).to(device=z.device, dtype=z.dtype)
                        offset[0] += b
                        return z + alpha_v * d
                    return _gen

                # Each config needs a fresh chunk-local offset; use a wrapper loop rather than closure state reset.
                rows: List[dict] = []
                l2_all: List[float] = []
                t0 = time.time()
                for i in range(0, len(tasks), BATCH_SIZE):
                    chunk = tasks[i : i + BATCH_SIZE]
                    z, s, t, ids = batch_tensors(chunk, src_latents, device)
                    d = torch.stack(diffs[i : i + len(chunk)], dim=0).to(device=device, dtype=z.dtype)
                    z_gen = z + alpha * d
                    l2 = (z_gen.float() - z.float()).flatten(1).norm(p=2, dim=1).cpu().tolist()
                    l2_all.extend(l2)
                    rows.extend(compute_batch_rows(z_gen, chunk, models, ref_feats, vae, trt, source_cache, latent_l2=l2))
                wall = time.time() - t0

                agg = aggregate_rows(rows)
                agg.update({
                    "NFE": 2 * n_noise,
                    "gen_wall_sec": round(wall, 1),
                    "mean_latent_l2": round(float(np.mean(l2_all)), 4),
                    "alpha": round(float(alpha), 4),
                    "alpha_multiplier": mult,
                    "sigma": sigma,
                    "n_noise": n_noise,
                    "ukiyo_dT_rel_VAE_Noop": round(float(agg["ukiyo_dT"] - noop_ukiyo), 4),
                    "dT_rel_VAE_Noop": round(float(agg["mean_dT"] - noop_agg["mean_dT"]), 4),
                })
                results["tweedie"][cfg] = agg
                print(f"  [{cfg:38s}] dT={agg['mean_dT']:+.4f} Top-1={agg['Target_Top1']:4.1f}% "
                      f"Margin={agg['Target_Margin']:+.4f} DINO-C={agg['DINO_C']:.4f} LPIPS={agg['LPIPS']:.4f} "
                      f"ukiyo={agg['ukiyo_top1']:4.1f}% ukiyo_dT_rel_noop={agg['ukiyo_dT_rel_VAE_Noop']:+.4f}", flush=True)

    # 5. Select best Tweedie by Top-1 with a light DINO-C guardrail (diagnostic only)
    eligible = [
        (name, d) for name, d in results["tweedie"].items()
        if d["DINO_C"] >= 0.60
    ]
    best_name, best = max(
        eligible or list(results["tweedie"].items()),
        key=lambda kv: (kv[1]["Target_Top1"], kv[1]["mean_dT"]),
    )
    results["best"] = {
        "name": best_name,
        "metrics": best,
        "targets_vs_Calib_Balanced_s16": {
            "NFE_reduced": best["NFE"] < results["baselines"]["Calib_Balanced_s16"]["NFE"],
            "Top1_ge_baseline": best["Target_Top1"] >= results["baselines"]["Calib_Balanced_s16"]["Target_Top1"],
            "dT_ge_+0.08": best["mean_dT"] >= 0.08,
            "DINO_C_ge_0.68": best["DINO_C"] >= 0.68,
            "Ukiyo_Top1_ge_15": best["ukiyo_top1"] >= 15.0,
            "Ukiyo_dT_rel_noop_gt_0": best["ukiyo_dT_rel_VAE_Noop"] > 0.0,
        },
    }

    out_json = V7_ROOT / "p0_tweedie_scan.json"
    save_json(results, out_json)
    print(f"\n[V7 P0] Saved {out_json}", flush=True)

    # Markdown report
    b = results["baselines"]["Calib_Balanced_s16"]
    lines = []
    lines.append("# WEAVE V7 P0：Tweedie 单步 / 少步差分扫描报告\n")
    lines.append("> 目标：在不训练新映射的前提下，验证 16 步 Score Difference（96 NFE）是否可以被")
    lines.append("> 2~8 次 ScoreNet 前向的 Tweedie 条件后验均值差分替代。\n")
    lines.append("> **验证集**：`distinct5_512/train` 中严格排除冻结 Test600 源图后的独立 50 张源图，")
    lines.append("> 共 200 个跨风格任务；所有噪声由 `task_id+noise_idx` 哈希确定，与 batch/顺序无关。\n")

    lines.append("## 一、基线对照\n")
    lines.append("| 配置 | NFE | CSD $\\Delta T \\uparrow$ | Target Top-1 $\\uparrow$ | Target Margin $\\uparrow$ | DINO-C $\\uparrow$ | LPIPS $\\downarrow$ | Ukiyo-e Top-1 | Ukiyo-e $\\Delta T$ | 潜空间位移 L2 | 200任务生成耗时 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in results["baselines"].items():
        lines.append(f"| **{name}** | {d['NFE']} | **{d['mean_dT']:+.4f}** | **{d['Target_Top1']}%** | **{d['Target_Margin']:+.4f}** | "
                     f"{d['DINO_C']:.4f} | {d['LPIPS']:.4f} | {d['ukiyo_top1']}% | {d['ukiyo_dT']:+.4f} | "
                     f"{d['mean_latent_l2']:.4f} | {d['gen_wall_sec']}s |")

    lines.append("\n## 二、Tweedie 扫描结果（按 Top-1 降序）\n")
    lines.append("| 配置 | $\\sigma$ | K | NFE | $\\alpha$ | CSD $\\Delta T \\uparrow$ | Top-1 $\\uparrow$ | Margin $\\uparrow$ | DINO-C $\\uparrow$ | LPIPS $\\downarrow$ | Ukiyo Top-1 | Ukiyo $\\Delta T$ vs VAE No-op |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in sorted(results["tweedie"].items(), key=lambda kv: -kv[1]["Target_Top1"]):
        lines.append(f"| **{name}** | {d['sigma']:.2f} | {d['n_noise']} | **{d['NFE']}** | {d['alpha']:.3f} | "
                     f"**{d['mean_dT']:+.4f}** | **{d['Target_Top1']}%** | {d['Target_Margin']:+.4f} | "
                     f"{d['DINO_C']:.4f} | {d['LPIPS']:.4f} | {d['ukiyo_top1']}% | **{d['ukiyo_dT_rel_VAE_Noop']:+.4f}** |")

    lines.append("\n## 三、诊断结论\n")
    lines.append(f"- **独立验证集上 Calib_Balanced_s16**：Top-1 = **{b['Target_Top1']}%**，$\\Delta T$ = **{b['mean_dT']:+.4f}**，DINO-C = **{b['DINO_C']:.4f}**，Ukiyo-e Top-1 = **{b['ukiyo_top1']}%**。\n")
    lines.append(f"- **当前最佳 Tweedie 配置**：`{best_name}`，NFE = **{best['NFE']}**（相对 96 NFE 减少 **{96/best['NFE']:.1f}×** ScoreNet 调用），Top-1 = **{best['Target_Top1']}%**，$\\Delta T$ = **{best['mean_dT']:+.4f}**，DINO-C = **{best['DINO_C']:.4f}**，Ukiyo-e Top-1 = **{best['ukiyo_top1']}%**。\n")
    lines.append("- 若 2~8 NFE 的 Tweedie 已接近 16 步积分，则继续优化 Score Difference 积分器的收益很低；")
    lines.append("  若 Tweedie 明显落后，说明多步积分确实提供了超出单步后验均值差的非线性路径，但仍应转向训练 1-NFE 分布传输算子。\n")

    out_md = DOCS_V7 / "P0_TWEEDIE_SCAN_REPORT.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[V7 P0] Saved {out_md}", flush=True)


if __name__ == "__main__":
    main()
