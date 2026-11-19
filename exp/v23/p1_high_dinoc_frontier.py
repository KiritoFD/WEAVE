"""V23 方向校准 A (4090): 瞄准 DINO-C >= 0.612 高内容 Pareto 前沿（支配 StyleGallery 52.5% / 0.5604 / 0.6113）。

思路:
此前偏离前沿在于使用了低 DINO-C 基座 (Peak1500, 0.48-0.53) 与全强度纯风格外推 wp*(v_s0-v_00)。
在 F3_dual_fixed_orig 上我们观察到: 修复后的真双分支 cs_j（无 wp 毁内容外推）天然达到 DINO-S=0.5522 @ DINO-C=0.6096。
本实验在高内容基座 (Joint_T1_C1AB, T1, V20_CoreSLF, Joint_Pk_C1AB) 上测试两类高内容保真策略:
1. 修复后真双分支 cs_j + 高内容起点 (tau_min=0.36, tau_hi=0.72) + 增强内容下风格差 ws_j (ws_fig=5.5 / 6.5)
2. 衰减纯风格四分支 (wp 降至 0.25x, wc=1.8, tau_min=0.36, tau_hi=0.72) —— 既抑制 Step0 内容砸毁又保留少许跨画派辅助。

用法 (4090):
  python -m exp.v23.p1_high_dinoc_frontier --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from . import OUT, EVAL_SEED, ROOT, RawWriterV23, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v23, sanitize_cuda_env, save_json, utc, validate_raw_meta
from .p1_fast_anchor import compute_prism1_anchor_750
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from src.utils import load_s10_decoder

CKPTS = {
    "JT1": OUT / "joint_c1ab/slf_Joint_T1_C1AB.pt",
    "JPk": OUT / "joint_c1ab/slf_Joint_Pk_C1AB.pt",
    "T1":  ROOT / "runs/v22/tmatrix/slf_T1.pt",
    "V20": ROOT / "runs/v20/slf_flow/slf_core_v2.pt",
}


@torch.no_grad()
def run_point(core: PrismCore, ckpt_path: Path, ctx: dict, arm_name: str,
              anc_t: torch.Tensor, gate: GateConfig, cfg: CFGConfig, solver: dict, device: str = "cuda") -> dict:
    cfg_hash = config_hash({"arm": arm_name, "ckpt": file_sha(ckpt_path)[:12], "gate": str(gate), "cfg": str(cfg), "solver": solver})
    if (OUT / "raw" / arm_name / "meta.json").exists():
        old = read_json(OUT / "raw" / arm_name / "meta.json")
        if old.get("cfg_hash") == cfg_hash and validate_raw_meta(old):
            s = read_json(OUT / f"measured/{arm_name}/summary_v23.json")
            c = s["cross600"]
            return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                    "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "lpips": round(c["lpips"], 5)}

    tasks, z_src_750 = ctx["tasks"], ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    wr = RawWriterV23(arm_name, dec=ctx["dec"], device=device)
    t0 = time.time()

    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        z_gen, _ = core.integrate_flow(
            ctx["eps_750"][sl], anc_t[sl].to(device), ctx["y16_750"][sl], ctx["d16_750"][sl],
            tid_all[sl].to(device), gate=gate, cfg=cfg, solver=solver,
        )
        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), z_gen)
        wr.write_batch(z_out, tasks[sl])

    meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_path),
                       tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                       seed=EVAL_SEED, wall_sec=round(time.time() - t0, 1))
    assert validate_raw_meta(meta)
    s = run_measure_v23(arm_name, device=device)
    export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
    c = s["cross600"]
    return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
            "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
            "lpips": round(c["lpips"], 5), "same_top1": round(s["same150"]["csd_top1"], 5)}


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
    z_prism1 = compute_prism1_anchor_750(bundle, tasks, z_src_750, d16_750, device=args.device)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(args.device)
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750, "d16_750": d16_750,
           "eps_750": eps_750, "dec": dec}

    # 高内容求解器基准: tau 偏高起点 + 衰减 wp 或纯 cs_j 增强 ws_j
    s_csj_hiw55 = dict(D5_SOLVER, tau_min=0.34, tau_hi=0.70, ws_fig=5.5)
    s_csj_hiw65 = dict(D5_SOLVER, tau_min=0.36, tau_hi=0.72, ws_fig=6.5)
    s_lowwp_d6  = dict(D5_SOLVER, tau_min=0.36, tau_hi=0.72, wc_fig=1.85, wc_abs=1.50,
                       ws_fig=5.0, wsp_fig_late=0.8, wsp_abs=1.5, lscale_abs=0.45)

    specs = [
        ("JT1", "HiC_JT1_csj55_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="cs_j"), s_csj_hiw55),
        ("JT1", "HiC_JT1_csj65_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="cs_j"), s_csj_hiw65),
        ("JT1", "HiC_JT1_lowwp_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="full", rescale_phi=0.48), s_lowwp_d6),
        ("JT1", "HiC_JT1_lowwp_p1", z_prism1, GateConfig(mode="original"), CFGConfig(branches="full", rescale_phi=0.48), s_lowwp_d6),
        ("JPk", "HiC_JPk_csj65_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="cs_j"), s_csj_hiw65),
        ("JPk", "HiC_JPk_lowwp_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="full", rescale_phi=0.48), s_lowwp_d6),
        ("V20", "HiC_V20_csj65_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="cs_j"), s_csj_hiw65),
        ("V20", "HiC_V20_lowwp_t4", z_t4,     GateConfig(mode="original"), CFGConfig(branches="full", rescale_phi=0.48), s_lowwp_d6),
    ]

    results = []
    by_ck = {}
    for k, arm_name, anc, g, cfg, solv in specs:
        ck = CKPTS[k]
        if not ck.exists():
            print(f"[HiC] SKIP {arm_name} (missing {ck})", flush=True)
            continue
        by_ck.setdefault(k, []).append((arm_name, anc, g, cfg, solv))

    for k, items in by_ck.items():
        ck = CKPTS[k]
        core = PrismCore(bundle, bank, ck, device=args.device)
        for arm_name, anc, g, cfg, solv in items:
            results.append(run_point(core, ck, ctx, arm_name, anc, g, cfg, solv, device=args.device))
        del core
        torch.cuda.empty_cache()

    save_json(OUT / "p1_high_dinoc_frontier.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 高 DINO-C (>=0.61) 前沿搜索汇总 (对标 SG: 52.50% / D-S 0.5604 / D-C 0.6113) ===", flush=True)
    for r in results:
        flag = " [DOMINATES_SG!]" if (r["dino_c"] >= 0.610 and r["dino_s"] >= 0.560) else (" [D-C>=0.61]" if r["dino_c"] >= 0.610 else "")
        print(f"{r['arm']:20s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}{flag}", flush=True)


if __name__ == "__main__":
    main()
