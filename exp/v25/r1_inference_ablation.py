"""V25 Round 1 (4090): 无训练推理结构消融 —— P0-1 单模型 vs Hybrid、P0-2 CFG 分支、P1-3 锚点、P1-4 Gate。

落实 docs/19_pull_together/01_convergence_plan.md 修正:
- C1 (τ 截断): exp/v22/core.py 将 tau_v clamp 至 <=0.78 —— 曲线 B 只做 tau_min 起点上移 (tau_hi 固定 0.78),
  每臂记录实测 tau 分布 (mean/p50/max/触顶比例), tau_hi>0.78 不再作为独立档位;
- C4 (配方↔分支): full 四分支 CFG 仅在完整训过全分支的 slf_core_v2.pt 上评估;
- 非劣性 CI 判定在收尾分析执行, 本脚本产出 750 评测 + τ 分布 + 结果 JSON。

用法 (4090):
  python -m exp.v25.r1_inference_ablation --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch

from . import OUT, EVAL_SEED, ROOT, RawWriterV25, config_hash, export_raw_pngs, git_sha, read_json, run_measure_v25, sanitize_cuda_env, save_json, utc, validate_raw_meta
from exp.v22.core import CFGConfig, D5_SOLVER, D6_SOLVER, GateConfig, PrismCore, effective_gate_components
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v24.p0_m0_m2 import P1_CKPT, V20_CKPT, compute_prism1_latents
from src.utils import load_s10_decoder

DUAL_SWD_CKPT = ROOT / "runs/v23/dual_swd/slf_DualSWD_pure.pt"
V24_BASE_CACHE = ROOT / "runs/v24/cache/p0_base_latents_750.pt"
D6H_SOLVER = dict(D6_SOLVER, tau_min=0.48, tau_hi=0.78)


@torch.no_grad()
def run_core_flow(bundle: dict, bank: dict, ckpt, tasks: list[dict], z_src_750: torch.Tensor,
                  anchor: torch.Tensor | None, y16_750: torch.Tensor, d16_750: torch.Tensor,
                  eps_750: torch.Tensor, branches: str = "full", gate_mode: str = "original",
                  solver: dict | None = None, device: str = "cuda", batch: int = 25):
    """通用 PrismCore 流积分 (可换权重/分支/Gate/solver), 同时按 integrate_flow 内部公式记录实测 τ 分布。"""
    solver = dict(solver)
    core = PrismCore(bundle, bank, ckpt, device=device)
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    gate = GateConfig(mode=gate_mode)
    cfg = CFGConfig(branches=branches, rescale_phi=0.48)
    out = torch.zeros_like(z_src_750, dtype=torch.float32)
    tau_rec = []
    for st in range(0, 750, batch):
        sl = slice(st, min(st + batch, 750))
        tid_d = tid_all[sl].to(device)
        # τ 分布记录: 与 integrate_flow 内部 tau_v 公式逐位一致 (build_style_condition 确定性匹配)
        _, dsd_ref = core.build_style_condition(d16_750[sl].to(device).float(), tid_d, mode="core8")
        g_tau, _, _, _ = effective_gate_components(dsd_ref, gate, float(core.bank["mid_dsd"]))
        tau_rec.append(((solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)).cpu())
        z_gen, _ = core.integrate_flow(
            eps_750[sl], None if anchor is None else anchor[sl].to(device), y16_750[sl], d16_750[sl], tid_d,
            gate=gate, cfg=cfg, solver=solver,
        )
        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), z_gen)
        out[sl] = z_out.cpu()
    tau = torch.cat(tau_rec)
    del core
    torch.cuda.empty_cache()
    stats = {
        "tau_cfg": [solver["tau_min"], solver["tau_hi"]],
        "tau_mean": round(float(tau.mean()), 4), "tau_p50": round(float(tau.median()), 4),
        "tau_max": round(float(tau.max()), 4),
        "frac_at_cap": round(float((tau >= 0.7799).float().mean()), 4),  # 被 0.78 clamp 触顶样本比例
        "branches": branches, "gate": gate_mode,
    }
    return out, stats


def eval_latent_v25(arm_name: str, z_750: torch.Tensor, tasks: list[dict], dec, device: str = "cuda") -> dict:
    cfg_hash = config_hash({"arm": arm_name, "v": 25})
    if (OUT / "raw" / arm_name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / arm_name / "meta.json")):
        s = read_json(OUT / f"measured/{arm_name}/summary_v25.json")
    else:
        wr = RawWriterV25(arm_name, dec=dec, device=device)
        t0 = time.time()
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            wr.write_batch(z_750[sl].to(device), tasks[sl])
        meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), wall_sec=round(time.time() - t0, 1))
        assert validate_raw_meta(meta)
        s = run_measure_v25(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
    c = s["cross600"]
    return {
        "arm": arm_name, "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
        "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5),
        "csd_top1": round(c["csd_top1"], 5), "same_top1": round(s["same150"]["csd_top1"], 5),
    }


def reused_v24_arm(arm_name: str) -> dict:
    s = read_json(ROOT / f"runs/v24/measured/{arm_name}/summary_v24.json")
    c = s["cross600"]
    return {
        "arm": arm_name, "reused": "runs/v24", "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
        "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5),
        "csd_top1": round(c["csd_top1"], 5), "same_top1": round(s["same150"]["csd_top1"], 5),
    }


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

    # V24 缓存基座 (z_p1_075/z_p1_100/z_d6 均为 T4 锚点 + full CFG 口径)
    d = torch.load(V24_BASE_CACHE, map_location="cpu", weights_only=False)
    z_p1_100, z_d6 = d["z_p1_100"], d["z_d6"]

    results, tau_records = [], {}
    out_json = OUT / "r1_inference_ablation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def emit(rec: dict, tau_stats: dict | None = None):
        results.append(rec)
        if tau_stats is not None:
            tau_records[rec["arm"]] = tau_stats
        save_json(out_json, {"updated_utc": utc(), "results": results, "tau_records": tau_records})

    # 复用 V24 已测基线 (零成本)
    for arm in ("M0_prism1_l075", "M0_prism1_l100", "M0_v20_d6", "M1_lin_g25", "M1_lin_g40", "M1_lin_g55"):
        emit(reused_v24_arm(arm))

    # ---- P0-1 曲线 A: PRISM-1 λ 起点上移 (单步前馈, 秒级) ----
    for lam in (1.30, 1.60):
        z = compute_prism1_latents(bundle, tasks, z_src_750, d16_750, lam, device=args.device)
        emit(eval_latent_v25(f"A_p1_l{int(lam*100)}", z, tasks, dec, device=args.device))

    # ---- P0-1 曲线 B: D6 tau_min 起点上移 (tau_hi 固定 0.78, 记录 τ 分布) ----
    for tmin in (0.50, 0.58, 0.66):
        z, st = run_core_flow(bundle, bank, V20_CKPT, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750,
                              branches="full", solver=dict(D6_SOLVER, tau_min=tmin, tau_hi=0.78), device=args.device)
        emit(eval_latent_v25(f"B_d6_t0{int(tmin*100)}", z, tasks, dec, device=args.device), st)
    if DUAL_SWD_CKPT.exists():  # 附跑: DualSWD 双分支高内容对照
        z, st = run_core_flow(bundle, bank, DUAL_SWD_CKPT, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750,
                              branches="cs_j", solver=dict(D6_SOLVER, tau_min=0.58, tau_hi=0.78, ws_fig=5.2), device=args.device)
        emit(eval_latent_v25("B_dswd_t058", z, tasks, dec, device=args.device), st)

    # ---- P0-1 曲线 C: Hybrid γ 上探 ----
    for gamma in (0.70, 0.85):
        emit(eval_latent_v25(f"C_hyb_g{int(gamma*100)}", (1.0 - gamma) * z_p1_100 + gamma * z_d6, tasks, dec, device=args.device))

    # ---- P0-2 CFG 分支 (slf_core_v2 + T4 锚点 + original Gate; C4: full 仅在此权重评估) ----
    p02 = [
        ("P02_d5_sconly", "sc_only", dict(D5_SOLVER)),
        ("P02_d5_csj_ws55", "cs_j", dict(D5_SOLVER, ws_fig=5.5)),
        ("P02_d5_csj_ws42", "cs_j", dict(D5_SOLVER)),
        ("P02_d5_full", "full", dict(D5_SOLVER)),
        ("P02_d6h_sconly", "sc_only", D6H_SOLVER),
        ("P02_d6h_csj", "cs_j", D6H_SOLVER),
        ("P02_d6h_full", "full", D6H_SOLVER),
    ]
    for arm, br, sv in p02:
        z, st = run_core_flow(bundle, bank, V20_CKPT, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750,
                              branches=br, solver=sv, device=args.device)
        emit(eval_latent_v25(arm, z, tasks, dec, device=args.device), st)

    # ---- P1-3 锚点 (slf_core_v2 + cs_j + D6-high; Anc_T4 参照 = P02_d6h_csj) ----
    p13 = [
        ("P13_ancp1_t040", z_p1_100, dict(D6H_SOLVER, tau_min=0.40)),
        ("P13_ancp1_t032", z_p1_100, dict(D6H_SOLVER, tau_min=0.32)),
        ("P13_ancsrc_t040", z_src_750, dict(D6H_SOLVER, tau_min=0.40)),
        ("P13_ancsrc_t032", z_src_750, dict(D6H_SOLVER, tau_min=0.32)),
    ]
    for arm, anc, sv in p13:
        z, st = run_core_flow(bundle, bank, V20_CKPT, tasks, z_src_750, anc, y16_750, d16_750, eps_750,
                              branches="cs_j", solver=sv, device=args.device)
        emit(eval_latent_v25(arm, z, tasks, dec, device=args.device), st)

    # ---- P1-4 Gate (slf_core_v2 + T4 锚点 + cs_j; original 对照 = P02_d5_csj_ws42 / P02_d6h_csj) ----
    for arm, sv in (("P14_d5_fixed_g50", dict(D5_SOLVER)), ("P14_d6h_fixed_g50", D6H_SOLVER)):
        z, st = run_core_flow(bundle, bank, V20_CKPT, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750,
                              branches="cs_j", gate_mode="fixed", solver=sv, device=args.device)
        emit(eval_latent_v25(arm, z, tasks, dec, device=args.device), st)

    print(f"\n=== V25 Round 1 完成: {len(results)} 臂 (τ 记录 {len(tau_records)} 条) -> {out_json} ===", flush=True)
    for r in results:
        tag = "(v24)" if r.get("reused") else "      "
        print(f"{r['arm']:20s}{tag}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
