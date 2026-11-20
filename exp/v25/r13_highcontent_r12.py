"""V25 R13 (4090): PRISM-HighContent-R12 严格对照消融与 γ 扫描.

动机 (2026-11-20):
  R12 锚一致重评中, MechA_B4anc_A0_global_g55 (0.45·z_P1 + 0.55·z_D6[B4,cs_j,D6H]) 达到:
    DINO-C=0.7723, DINO-S=0.4564, LPIPS=0.4783, Top-1=42.50%
  四项点估计全部超过 PRISM-1 λ=0.75 (0.7715 / 0.4371 / 0.4849 / 37.67%),
  且内容/风格/LPIPS 均优于旧 Hybrid γ=0.25 (0.7527 / 0.4481 / 0.4988 / 47.50%).

本脚本严格控制随机种子 (统一 EVAL_SEED=20261185) 并做两件事:
  1. PRISM-HighContent-R12 (B4 + cs_j + D6H) 完整 γ 扫描:
     γ ∈ {0.40, 0.55(复用 MechA_B4anc_A0_global_g55), 0.70, 0.85, 1.00}
     → 测出它在 DINO-C ∈ [0.72, 0.78] 的完整 Pareto 曲线, 直接在匹配内容水位下与旧 Hybrid / PRISM-1 比对.
  2. γ=0.55 严格三因子解耦 (Anchor ∈ {T4, B4} × CFG ∈ {full, cs_j} × Solver ∈ {D6, D6H}):
     - HC_B4_full_D6_g55  (B4 + full + D6)  → 与 M1_lin_g55 (T4 + full + D6) 单变量隔离 Anchor 效应
     - HC_B4_csj_D6_g55   (B4 + cs_j + D6)  → 隔离 CFG 效应 (vs full_D6) 与 Solver 效应 (vs csj_D6H)
     - HC_B4_full_D6H_g55 (B4 + full + D6H) → 隔离 Solver 效应 (vs full_D6) 与 CFG 效应 (vs csj_D6H)
     - HC_B4_full_D6_g100 (纯单模型 B4 + full + D6, γ=1.00)
  3. 对齐 A01 源图聚类配对 Bootstrap (n_boot=4000, seed=2026), 输出正式 CI 与因子分解增量表.

用法 (4090):
  python -m exp.v25.r13_highcontent_r12 --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

from . import EVAL_SEED, OUT, ROOT, sanitize_cuda_env, save_json, utc
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v22.core import D6_SOLVER
from exp.v24.p0_m0_m2 import V20_CKPT
from exp.v25.r1_inference_ablation import D6H_SOLVER, eval_latent_v25, reused_v24_arm, run_core_flow
from exp.v25.r12_anchor_reeval import build_or_load_ctx750_b4
from src.utils import load_s10_decoder

OUT_JSON = OUT / "r13_highcontent_r12.json"
SWEEP_GAMMAS = (0.40, 0.70, 0.85, 1.00)  # 0.55 已有 MechA_B4anc_A0_global_g55


def mix_latents(z_p1: torch.Tensor, z_d6: torch.Tensor, gamma: float,
                same_mask: torch.Tensor, z_src: torch.Tensor) -> torch.Tensor:
    """线性混合 + 网络外 same-task bypass (保证 same150 恒等语义)."""
    z_mix = (1.0 - float(gamma)) * z_p1.float() + float(gamma) * z_d6.float()
    m = same_mask.bool()
    if m.any():
        z_mix = z_mix.clone()
        z_mix[m] = z_src.float()[m]
    return z_mix


def _summarize_v25_arm(arm: str) -> dict:
    from . import read_json
    s = read_json(OUT / "measured" / arm / "summary_v25.json")
    c = s["cross600"]
    return {
        "arm": arm,
        "reused": "runs/v25",
        "dino_c": round(c["dino_c"], 5),
        "dino_s": round(c["dino_s"], 5),
        "lpips": round(c["lpips"], 5),
        "style_affinity": round(c["style_affinity"], 5),
        "csd_top1": round(c["csd_top1"], 5),
        "same_top1": round(s["same150"]["csd_top1"], 5),
    }


def compute_paired_cis(n_boot: int = 4000, seed: int = 2026) -> dict:
    from tools.v25_cpu.a01_noninferiority import compare_arms

    specs = [
        # (key, arm_a, ver_a, arm_b, ver_b, note)
        ("r12_g55_vs_prism1_l075", "MechA_B4anc_A0_global_g55", "v25", "M0_prism1_l075", "v24",
         "Pareto check @ DINO-C~0.772"),
        ("r12_g70_vs_hyb_g25", "HC_R12_g70", "v25", "M1_lin_g25", "v24",
         "High-content hybrid comparison @ ~0.75-0.76"),
        ("r12_g70_vs_hyb_g40", "HC_R12_g70", "v25", "M1_lin_g40", "v24",
         "High-content hybrid comparison @ ~0.747-0.76"),
        ("r12_g85_vs_hyb_g55", "HC_R12_g85", "v25", "M1_lin_g55", "v24",
         "Balanced-high comparison @ ~0.73-0.74"),
        ("r12_g100_vs_hyb_g55", "HC_R12_g100", "v25", "M1_lin_g55", "v24",
         "Single-model B4+csj+D6H vs old Hybrid g55"),
        ("factor_anchor_B4_vs_T4_under_full_D6", "HC_B4_full_D6_g55", "v25", "M1_lin_g55", "v24",
         "Pure Anchor effect (B4 vs T4) at fixed full+D6+g55"),
        ("factor_anchor_B4_vs_T4_under_csj_D6H", "MechA_B4anc_A0_global_g55", "v25", "MechA_A0_global_g55", "v25",
         "Pure Anchor effect (B4 vs T4) at fixed csj+D6H+g55"),
        ("factor_cfg_csj_vs_full_under_B4_D6", "HC_B4_csj_D6_g55", "v25", "HC_B4_full_D6_g55", "v25",
         "Pure CFG effect (cs_j vs full) under B4+D6+g55"),
        ("factor_cfg_csj_vs_full_under_B4_D6H", "MechA_B4anc_A0_global_g55", "v25", "HC_B4_full_D6H_g55", "v25",
         "Pure CFG effect (cs_j vs full) under B4+D6H+g55"),
        ("factor_solver_D6H_vs_D6_under_B4_csj", "MechA_B4anc_A0_global_g55", "v25", "HC_B4_csj_D6_g55", "v25",
         "Pure Solver effect (D6H vs D6) under B4+csj+g55"),
        ("factor_solver_D6H_vs_D6_under_B4_full", "HC_B4_full_D6H_g55", "v25", "HC_B4_full_D6_g55", "v25",
         "Pure Solver effect (D6H vs D6) under B4+full+g55"),
    ]
    out = {}
    for key, a, va, b, vb, note in specs:
        try:
            r = compare_arms(a, b, version_simple=va, version_complex=vb, n_boot=n_boot, seed=seed)
            out[key] = {"note": note, "arm_a": a, "arm_b": b, "final": r["final"],
                        "ci": r["ci"], "guards": r["guards"]}
        except Exception as exc:
            out[key] = {"note": note, "arm_a": a, "arm_b": b, "error": str(exc)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    ctx, z_anc_b4 = build_or_load_ctx750_b4(bundle, bank, args.device)
    tasks = ctx["tasks"]
    z_src, z_p1, z_d6_b4_csj_d6h = ctx["z_src"], ctx["z_p1"], ctx["z_d6"]
    same = ctx["same"]
    dec = ctx["dec"]

    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    results = {}
    tau_records = {"B4_csj_D6H": ctx["tau_stats"]}

    def save_state():
        save_json(OUT_JSON, {
            "task": "V25-R13 PRISM-HighContent-R12 controlled ablation",
            "updated_utc": utc(),
            "eval_seed": EVAL_SEED,
            "tau_records": tau_records,
            "results": results,
        })

    # Reused references
    for arm in ("M0_prism1_l075", "M0_prism1_l100", "M1_lin_g25", "M1_lin_g40", "M1_lin_g55"):
        results[arm] = reused_v24_arm(arm)
    for arm in ("MechA_A0_global_g55", "MechA_B4anc_A0_global_g55"):
        if (OUT / "measured" / arm / "summary_v25.json").exists():
            results[arm] = _summarize_v25_arm(arm)
    save_state()

    # 1. PRISM-HighContent-R12 (B4 + cs_j + D6H) gamma sweep (zero flow cost, reuses ctx750_B4!)
    for g in SWEEP_GAMMAS:
        arm = f"HC_R12_g{int(round(g * 100)):02d}" if g < 1.0 else "HC_R12_g100"
        z_out = mix_latents(z_p1, z_d6_b4_csj_d6h, g, same, z_src)
        results[arm] = eval_latent_v25(arm, z_out, tasks, dec, device=args.device)
        results[arm]["cfg"] = {"anchor": "B4", "branches": "cs_j", "solver": "D6H", "gamma": g}
        save_state()

    # 2. 3-factor disentanglement at gamma=0.55 (+ pure B4+full+D6 at gamma=1.00)
    flow_configs = [
        ("B4_full_D6", "full", dict(D6_SOLVER), [(0.55, "HC_B4_full_D6_g55"), (1.00, "HC_B4_full_D6_g100")]),
        ("B4_csj_D6", "cs_j", dict(D6_SOLVER), [(0.55, "HC_B4_csj_D6_g55")]),
        ("B4_full_D6H", "full", dict(D6H_SOLVER), [(0.55, "HC_B4_full_D6H_g55")]),
    ]
    for tag, br, sv, mix_list in flow_configs:
        z_flow, st = run_core_flow(
            bundle, bank, V20_CKPT, tasks, z_src, z_anc_b4, y16_750, d16_750, eps_750,
            branches=br, gate_mode="original", solver=sv, device=args.device,
        )
        tau_records[tag] = st
        for g, arm in mix_list:
            z_out = mix_latents(z_p1, z_flow, g, same, z_src)
            results[arm] = eval_latent_v25(arm, z_out, tasks, dec, device=args.device)
            results[arm]["cfg"] = {"anchor": "B4", "branches": br, "solver": tag.split("_")[-1], "gamma": g}
            save_state()

    # 3. Paired Bootstrap CIs
    paired_ci = compute_paired_cis(n_boot=args.n_boot, seed=2026)
    save_json(OUT_JSON, {
        "task": "V25-R13 PRISM-HighContent-R12 controlled ablation",
        "updated_utc": utc(),
        "eval_seed": EVAL_SEED,
        "tau_records": tau_records,
        "results": results,
        "paired_ci": paired_ci,
    })
    print("\n=== V25 R13 PRISM-HighContent-R12 汇总 ===", flush=True)
    for k, r in results.items():
        print(f"  {k:28s}: D-C={r['dino_c']:.4f} D-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
