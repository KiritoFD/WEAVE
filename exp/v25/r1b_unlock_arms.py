"""V25 R1b (4090, ~5 min): A01 CI 审计解锁臂 —— 补齐三个判定缺口的真实工作点。

1. `B_d6_t070 / B_d6_t072`: τ_min 起点上移到 0.70/0.72 → D6 真实覆盖 DINO-C≈0.71–0.74,
   使 "D6 vs Hybrid @ 0.73" 从未决变为可配对判定 (禁止外推伪造配对 —— A01 验收第 5 条);
2. `P14b_d6h_fixed_g69`: fixed 门控常数重标定 (0.5 → 0.69, 对齐 original 在 D6H 的 mean τ=0.689),
   排除 A01 报告的 "fixed τ 重分布混杂" 后重判密度门控可删性;
3. `P13b_ancp1_t044`: PRISM-1 锚 τ_min=0.44 (内容水位 ≈0.674, 对齐 T4 臂 0.6782),
   消除 P13 的 τ 混杂后重判锚点替代。

用法 (4090): python -m exp.v25.r1b_unlock_arms --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch

from . import OUT, EVAL_SEED, ROOT, sanitize_cuda_env, save_json, utc
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v22.core import D6_SOLVER
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v24.p0_m0_m2 import V20_CKPT
from exp.v25.r1_inference_ablation import D6H_SOLVER, V24_BASE_CACHE, eval_latent_v25, run_core_flow
from src.utils import load_s10_decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(args.device)
    z_p1_100 = torch.load(V24_BASE_CACHE, map_location="cpu", weights_only=False)["z_p1_100"]

    arms = [
        # (name, anchor, branches, gate_mode, fixed_g, solver)
        ("B_d6_t070", z_t4, "full", "original", 0.5, dict(D6_SOLVER, tau_min=0.70, tau_hi=0.78)),
        ("B_d6_t072", z_t4, "full", "original", 0.5, dict(D6_SOLVER, tau_min=0.72, tau_hi=0.78)),
        ("P14b_d6h_fixed_g69", z_t4, "cs_j", "fixed", 0.69, dict(D6H_SOLVER)),
        ("P13b_ancp1_t044", z_p1_100, "cs_j", "original", 0.5, dict(D6H_SOLVER, tau_min=0.44)),
    ]
    results, tau_records = [], {}
    out_json = OUT / "r1b_unlock_arms.json"
    for name, anc, br, gm, fg, sv in arms:
        z, st = run_core_flow(bundle, bank, V20_CKPT, tasks, z_src_750, anc, y16_750, d16_750, eps_750,
                              branches=br, gate_mode=gm, fixed_g=fg, solver=sv, device=args.device)
        rec = eval_latent_v25(name, z, tasks, dec, device=args.device)
        results.append(rec)
        tau_records[name] = st
        save_json(out_json, {"updated_utc": utc(), "results": results, "tau_records": tau_records})
        print(f"[V25 R1b] {name}: D-C={rec['dino_c']:.4f} D-S={rec['dino_s']:.4f} LPIPS={rec['lpips']:.4f} Top1={rec['csd_top1']*100:.2f}% tau_mean={st['tau_mean']}", flush=True)

    print(f"\n=== V25 R1b 完成: {len(results)} 臂 -> {out_json} ===", flush=True)


if __name__ == "__main__":
    main()
