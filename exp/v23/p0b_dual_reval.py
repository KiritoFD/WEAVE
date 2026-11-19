"""V23 P0-B (4090): 双分支修复对照 F0/F1/F4/F5（同 750 任务、同噪声、同 T4 锚点、不重训）。

实验臂:
- F0_v20_buggy_orig : 复核/读取 V20 Core-SLF 原错误双分支 (l2=[lscale, 0], original gate)
- F0_v20_buggy_g50  : V20 Core-SLF 原错误双分支 (fixed g=0.5)
- F1_v20_fixed_orig : V20 Core-SLF 修复后真双分支 (l2=[lscale, lscale], original gate)
- F1_v20_fixed_g50  : V20 Core-SLF 修复后真双分支 (fixed g=0.5)
- F4_pk_fixed_g50   : V22 Peak1500 修复后真双分支 (fixed g=0.5)
- F4_pk_fixed_g45   : V22 Peak1500 修复后真双分支 (fixed g=0.45)
- F4_pk_fixed_g45_hiw : V22 Peak1500 修复后真双分支高指导 (fixed g=0.45, ws_fig=6.0)
- F5_pk_full_g45    : 复用 V22 Peak1500 四分支强基线 (T4Pg45_wc10 = 76.83%)

用法 (4090):
  python -m exp.v23.p0b_dual_reval --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch

from . import OUT, EVAL_SEED, ROOT, RawWriterV23, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v23, sanitize_cuda_env, save_json, utc, validate_raw_meta
from exp.v22.core import (
    CFGConfig,
    D5_SOLVER,
    GateConfig,
    PrismCore,
    _guidance_curves,
    build_content_condition,
    effective_gate_components,
    make_initial_state,
    rk2_integrate,
)
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank

V20_CKPT = ROOT / "runs/v20/slf_flow/slf_core_v2.pt"
PEAK_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"


def _predict_buggy_csj(model, z_cur, tv, content, tid, u_style, lscale, g_guid, solver, prog):
    """复刻修复前错误实现: 第二分支 layout_scale 被清空为 0。"""
    B = z_cur.shape[0]
    dev = z_cur.device
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls0 = torch.zeros(B, device=dev)
    z2 = torch.cat([z_cur, z_cur], 0)
    y2 = content["y_in"].repeat(2, 1, 1, 1)
    d2 = content["d_in"].repeat(2, 1, 1, 1)
    s2 = tid.repeat(2)
    u2 = u_style.repeat(2, 1)
    t2 = tv.repeat(2)
    ds2 = torch.cat([f0, f1], 0)
    dc2 = torch.cat([f0, f0], 0)
    l2 = torch.cat([lscale, ls0], 0)
    with autocast(str(dev)):
        v_sc, v_0c = model(z2, y2, d2, s2, u2, t2, drop_style=ds2, drop_content=dc2, layout_scale=l2).float().chunk(2, 0)
    _, ws_j, _ = _guidance_curves(g_guid, solver, prog)
    return v_0c + ws_j * (v_sc - v_0c)


@torch.no_grad()
def run_arm(core: PrismCore, ckpt_path, ctx: dict, arm_name: str, gate: GateConfig,
            buggy_ls0: bool = False, solver_over: dict | None = None, device: str = "cuda") -> dict:
    cfg_hash = config_hash({"arm": arm_name, "ckpt": file_sha(ckpt_path)[:12], "gate": str(gate),
                            "buggy": buggy_ls0, "solver_over": solver_over})
    out_dir = OUT / "raw" / arm_name
    if out_dir.joinpath("meta.json").exists():
        old = read_json(out_dir / "meta.json")
        if old.get("cfg_hash") == cfg_hash and validate_raw_meta(old):
            s = read_json(OUT / f"measured/{arm_name}/summary_v23.json")
            c = s["cross600"]
            print(f"[V23 SKIP] {arm_name}: Top1={c['csd_top1']*100:.2f}% D-C={c['dino_c']:.4f}", flush=True)
            return {"arm": arm_name, **c}

    tasks = ctx["tasks"]
    z_src_750 = ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    anchor_750 = ctx["z_t4"]
    eps_750 = ctx["eps_750"]
    y16_750, d16_750 = ctx["y16_750"], ctx["d16_750"]
    dec = ctx["dec"]

    solver = dict(D5_SOLVER)
    if solver_over:
        solver.update(solver_over)
    cfg = CFGConfig(branches="cs_j", rescale_phi=0.48)
    wr = RawWriterV23(arm_name, dec=dec, device=device)
    t0 = time.time()

    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        grp = tasks[sl]
        tid_b = tid_all[sl].to(device)
        if not buggy_ls0:
            z_gen, _ = core.integrate_flow(
                eps_750[sl], anchor_750[sl].to(device), y16_750[sl], d16_750[sl], tid_b,
                gate=gate, cfg=cfg, solver=solver,
            )
        else:
            d16_b = d16_750[sl].to(device).float()
            y16_b = y16_750[sl].to(device).float()
            u_style, dsd_ref = core.build_style_condition(d16_b, tid_b, mode="core8")
            g_tau, g_layout, g_guid, _ = effective_gate_components(dsd_ref, gate, float(core.bank["mid_dsd"]))
            lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
            content = build_content_condition(y16_b, d16_b, g_layout)
            tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
            z = make_initial_state(eps_750[sl].to(device).float(), anchor_750[sl].to(device).float(), tau_v)
            z_gen = rk2_integrate(
                lambda zz, prog: _predict_buggy_csj(
                    core.model, zz, tau_v + prog * (1.0 - tau_v), content, tid_b, u_style,
                    lscale, g_guid, solver, prog,
                ),
                z, tau_v, int(solver["steps"]),
            )

        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device),
                            z_src_750[sl].to(device).float(), z_gen)
        wr.write_batch(z_out, grp)

    meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_path),
                       tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                       seed=EVAL_SEED, buggy_ls0=buggy_ls0, wall_sec=round(time.time() - t0, 1))
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
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    from src.utils import load_s10_decoder
    dec = load_s10_decoder(args.device)
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750, "d16_750": d16_750,
           "eps_750": eps_750, "z_t4": z_t4, "dec": dec}

    results = []
    # V20 Core-SLF: F0 (buggy) vs F1 (fixed)
    core_v20 = PrismCore(bundle, bank, V20_CKPT, device=args.device)
    results.append(run_arm(core_v20, V20_CKPT, ctx, "F0_v20_buggy_orig", GateConfig(mode="original"), buggy_ls0=True, device=args.device))
    results.append(run_arm(core_v20, V20_CKPT, ctx, "F0_v20_buggy_g50", GateConfig(mode="fixed", fixed_g=0.5), buggy_ls0=True, device=args.device))
    results.append(run_arm(core_v20, V20_CKPT, ctx, "F1_v20_fixed_orig", GateConfig(mode="original"), buggy_ls0=False, device=args.device))
    results.append(run_arm(core_v20, V20_CKPT, ctx, "F1_v20_fixed_g50", GateConfig(mode="fixed", fixed_g=0.5), buggy_ls0=False, device=args.device))
    del core_v20
    torch.cuda.empty_cache()

    # V22 Peak1500: F4 (fixed two-branch) vs F5 (four-branch strong baseline)
    core_pk = PrismCore(bundle, bank, PEAK_CKPT, device=args.device)
    results.append(run_arm(core_pk, PEAK_CKPT, ctx, "F4_pk_buggy_g50", GateConfig(mode="fixed", fixed_g=0.5), buggy_ls0=True, device=args.device))
    results.append(run_arm(core_pk, PEAK_CKPT, ctx, "F4_pk_fixed_g50", GateConfig(mode="fixed", fixed_g=0.5), buggy_ls0=False, device=args.device))
    results.append(run_arm(core_pk, PEAK_CKPT, ctx, "F4_pk_fixed_g45", GateConfig(mode="fixed", fixed_g=0.45), buggy_ls0=False, device=args.device))
    results.append(run_arm(core_pk, PEAK_CKPT, ctx, "F4_pk_fixed_g45_hiw", GateConfig(mode="fixed", fixed_g=0.45), buggy_ls0=False,
                           solver_over={"ws_fig": 6.0}, device=args.device))

    save_json(OUT / "p0b_dual_reval.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P0-B 双分支修复汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:24s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
