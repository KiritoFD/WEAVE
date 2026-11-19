"""V22 P3-N3 条件指导方向冲突控制（诊断 + 无训练推理消融）+ 完整成本/延迟 benchmark。

实验内容:
1. N3_diag: 测量 Peak1500 在推理过程中 d_s0=(v_s0-v_00), d_0c=(v_0c-v_00), d_sc=(v_sc-v_0c)
   的夹角余弦、负相关占比与 RMS 范数比例。
2. N3_neg_proj: 仅消除 d_s0 中与 d_0c 冲突(点积<0)的分量（d_s0 -= proj(d_s0, d_0c) if dot<0）。
3. N3_ortho: 完全正交化 d_s0 相对 d_0c（d_s0 -= proj(d_s0, d_0c)）。
4. N3_rms_cap: 限制 ||d_s0|| <= 1.2 * ||d_0c||。
5. Cost Benchmark: B=1 / B=16 推理延迟 (ms/img)、GPU 峰值显存、参数量、NFE。

用法 (4090):
  python -m exp.v22.eval_n3_conflict --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch

from . import OUT, EVAL_SEED, ROOT, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v22, sanitize_cuda_env, save_json, utc, validate_raw_meta
from .core import (
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
from .t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank

PEAK_CKPT = OUT / "tmatrix" / "slf_T4Ssnap1500.pt"


def _cos_and_rms(a: torch.Tensor, b: torch.Tensor):
    af, bf = a.flatten(1), b.flatten(1)
    an = af.norm(dim=1).clamp_min(1e-8)
    bn = bf.norm(dim=1).clamp_min(1e-8)
    cos = (af * bf).sum(dim=1) / (an * bn)
    rms_ratio = an / bn
    return cos, rms_ratio


def predict_n3(
    model,
    z_cur: torch.Tensor,
    tv: torch.Tensor,
    content: dict,
    tid: torch.Tensor,
    u_style: torch.Tensor,
    lscale: torch.Tensor,
    g_guid: torch.Tensor,
    solver: dict,
    prog: float,
    mode: str,
    stats_buf: list | None = None,
) -> torch.Tensor:
    B = z_cur.shape[0]
    dev = z_cur.device
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls0 = torch.zeros(B, device=dev)

    z4 = z_cur.repeat(4, 1, 1, 1)
    y4 = content["y_in"].repeat(4, 1, 1, 1)
    d4 = content["d_in"].repeat(4, 1, 1, 1)
    s4 = tid.repeat(4)
    u4 = u_style.repeat(4, 1)
    t4 = tv.repeat(4)
    ds = torch.cat([f0, f0, f1, f1], 0)
    dc = torch.cat([f0, f1, f0, f1], 0)
    lsc4 = torch.cat([lscale, ls0, lscale, ls0], 0)
    with autocast(str(dev)):
        v_sc, v_s0, v_0c, v_00 = model(
            z4, y4, d4, s4, u4, t4, drop_style=ds, drop_content=dc, layout_scale=lsc4
        ).float().chunk(4, 0)

    d_0c = v_0c - v_00
    d_sc = v_sc - v_0c
    d_s0 = v_s0 - v_00

    if stats_buf is not None:
        c_s0_0c, r_s0_0c = _cos_and_rms(d_s0, d_0c)
        c_sc_0c, r_sc_0c = _cos_and_rms(d_sc, d_0c)
        stats_buf.append({
            "cos_s0_0c": float(c_s0_0c.mean()),
            "neg_frac_s0_0c": float((c_s0_0c < 0).float().mean()),
            "rms_ratio_s0_0c": float(r_s0_0c.mean()),
            "cos_sc_0c": float(c_sc_0c.mean()),
            "rms_ratio_sc_0c": float(r_sc_0c.mean()),
        })

    if mode in ("neg_proj", "ortho"):
        a = d_s0.flatten(1)
        b = d_0c.flatten(1)
        dot = (a * b).sum(dim=1, keepdim=True)
        bb = (b * b).sum(dim=1, keepdim=True).clamp_min(1e-8)
        proj = (dot / bb) * b
        if mode == "neg_proj":
            mask = (dot < 0).float()
            a = a - mask * proj
        else:
            a = a - proj
        d_s0 = a.view_as(d_s0)
    elif mode == "rms_cap":
        an = d_s0.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
        bn = d_0c.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
        cap = 1.2 * bn
        scale = torch.minimum(torch.ones_like(an), cap / an).view(B, 1, 1, 1)
        d_s0 = d_s0 * scale

    wc, ws_j, ws_p = _guidance_curves(g_guid, solver, prog)
    vr = v_00 + wc * d_0c + ws_j * d_sc + ws_p * d_s0
    phi = float(solver.get("rescale_phi", 0.48))
    if phi > 0.0:
        gf4 = g_guid.view(B, 1, 1, 1)
        s_ref = (gf4 * v_sc + (1.0 - gf4) * v_s0).flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
        s_raw = vr.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
        vr = phi * vr * (s_ref / s_raw) + (1.0 - phi) * vr
    return vr


@torch.no_grad()
def run_n3_arm(core: PrismCore, ctx: dict, arm_name: str, mode: str, fixed_g: float, wc_val: float,
               collect_diag: bool = False, device: str = "cuda") -> tuple[dict, dict | None]:
    from src.utils.raw_io import RawWriter
    from src.utils import decode_s10

    tasks = ctx["tasks"]
    z_src_750 = ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    anchor_750 = ctx["z_t4"]
    eps_750 = ctx["eps_750"]
    y16_750, d16_750 = ctx["y16_750"], ctx["d16_750"]
    dec = ctx["dec"]

    cfg_hash = config_hash({"arm": arm_name, "mode": mode, "g": fixed_g, "wc": wc_val})
    out_dir = OUT / "raw" / arm_name
    diag_summary = None
    if out_dir.joinpath("meta.json").exists() and not collect_diag:
        old = read_json(out_dir / "meta.json")
        if old.get("cfg_hash") == cfg_hash and validate_raw_meta(old):
            s = read_json(OUT / f"measured/{arm_name}/summary_v22.json")
            c = s["cross600"]
            return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5),
                    "dino_s": round(c["dino_s"], 5), "dino_c": round(c["dino_c"], 5),
                    "style_affinity": round(c["style_affinity"], 5), "lpips": round(c["lpips"], 5)}, None

    gate = GateConfig(mode="fixed", fixed_g=fixed_g)
    solver = dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, steps=10,
                  ws_fig=4.2, wsp_fig_late=2.4, wsp_abs=5.2,
                  wc_fig=wc_val, wc_abs=wc_val, lscale_abs=0.34)
    wr = RawWriter(arm_name, raw_root=OUT / "raw", kind="v22")
    t0 = time.time()
    diag_buf: list[dict] | None = [] if collect_diag else None

    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        grp = tasks[sl]
        d16_b = d16_750[sl].to(device).float()
        y16_b = y16_750[sl].to(device).float()
        tid_b = tid_all[sl].to(device)
        u_style, dsd_ref = core.build_style_condition(d16_b, tid_b, mode="core8")
        g_tau, g_layout, g_guid, _ = effective_gate_components(dsd_ref, gate, float(core.bank["mid_dsd"]))
        lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
        content = build_content_condition(y16_b, d16_b, g_layout)
        tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
        z = make_initial_state(eps_750[sl].to(device).float(), anchor_750[sl].to(device).float(), tau_v)

        buf = diag_buf if (collect_diag and not same_all[sl].all()) else None
        z_gen = rk2_integrate(
            lambda zz, prog: predict_n3(
                core.model, zz, tau_v + prog * (1.0 - tau_v), content, tid_b, u_style,
                lscale, g_guid, solver, prog, mode, stats_buf=buf,
            ),
            z, tau_v, int(solver["steps"]),
        )
        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device),
                            z_src_750[sl].to(device).float(), z_gen)
        wr.append([t["task_id"] for t in grp],
                  decode_s10(dec, z_out, batch=16).clamp(0.0, 1.0), z_out.cpu())

    meta_p = wr.finish(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(PEAK_CKPT),
                       tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                       seed=EVAL_SEED, mode=mode, fixed_g=fixed_g, wc=wc_val,
                       wall_sec=round(time.time() - t0, 1))
    assert validate_raw_meta(read_json(meta_p / "meta.json"))
    run_measure_v22(arm_name, device=device)
    export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
    s = read_json(OUT / f"measured/{arm_name}/summary_v22.json")
    c = s["cross600"]
    if diag_buf:
        diag_summary = {
            k: round(sum(x[k] for x in diag_buf) / len(diag_buf), 4)
            for k in diag_buf[0].keys()
        }
    row = {
        "arm": arm_name, "mode": mode, "g": fixed_g, "wc": wc_val,
        "csd_top1": round(c["csd_top1"], 5),
        "dino_s": round(c["dino_s"], 5),
        "style_affinity": round(c["style_affinity"], 5),
        "dino_c": round(c["dino_c"], 5),
        "lpips": round(c["lpips"], 5),
    }
    print(f"[N3] {arm_name}: Top1={c['csd_top1']*100:.2f}% DINO-S={c['dino_s']:.4f} "
          f"DINO-C={c['dino_c']:.4f} Aff={c['style_affinity']:.4f} LPIPS={c['lpips']:.4f}", flush=True)
    return row, diag_summary


@torch.no_grad()
def benchmark_cost(core: PrismCore, ctx: dict, device: str = "cuda") -> dict:
    from src.utils import decode_s10

    dec = ctx["dec"]
    n_params_slf = sum(p.numel() for p in core.model.parameters())
    n_params_s10 = sum(p.numel() for p in dec.parameters())

    out = {"params_slf_M": round(n_params_slf / 1e6, 2), "params_s10_M": round(n_params_s10 / 1e6, 2)}
    gate = GateConfig(mode="fixed", fixed_g=0.45)
    cfg = CFGConfig(branches="full", rescale_phi=0.48)
    solver = dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, steps=10,
                  ws_fig=4.2, wsp_fig_late=2.4, wsp_abs=5.2,
                  wc_fig=1.0, wc_abs=1.0, lscale_abs=0.34)

    for bsz in (1, 16):
        eps = ctx["eps_750"][:bsz]
        anc = ctx["z_t4"][:bsz].to(device)
        y16 = ctx["y16_750"][:bsz]
        d16 = ctx["d16_750"][:bsz]
        tid = torch.zeros(bsz, dtype=torch.long, device=device)

        # warmup
        for _ in range(2):
            zg, _ = core.integrate_flow(eps, anc, y16, d16, tid, gate=gate, cfg=cfg, solver=solver)
            _ = decode_s10(dec, zg, batch=bsz)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        n_iter = 10 if bsz == 1 else 4
        t_flow = 0.0
        t_dec = 0.0
        for _ in range(n_iter):
            t0 = time.perf_counter()
            zg, _ = core.integrate_flow(eps, anc, y16, d16, tid, gate=gate, cfg=cfg, solver=solver)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            _ = decode_s10(dec, zg, batch=bsz)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t_flow += (t1 - t0)
            t_dec += (t2 - t1)

        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        flow_ms = t_flow * 1000.0 / (n_iter * bsz)
        dec_ms = t_dec * 1000.0 / (n_iter * bsz)
        out[f"b{bsz}_flow_ms_per_img"] = round(flow_ms, 2)
        out[f"b{bsz}_s10_ms_per_img"] = round(dec_ms, 2)
        out[f"b{bsz}_hot_total_ms_per_img"] = round(flow_ms + dec_ms, 2)
        out[f"b{bsz}_peak_vram_mb"] = round(peak_mb, 1)
    print(f"[COST] {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(OUT.parent.parent.parent / "pseudo_gt_24h/orig_mean_fp16"))
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

    core = PrismCore(bundle, bank, PEAK_CKPT, device=args.device)
    # 先跑 N3 诊断（顺便验证与 T4Pg45_wc10 一致）+ 3 个冲突控制臂 (g=0.45, wc=1.0)
    rows = []
    _, diag_g45 = run_n3_arm(core, ctx, "N3_diag_g45_wc10", "none", 0.45, 1.0, collect_diag=True, device=args.device)
    r_neg, _ = run_n3_arm(core, ctx, "N3a_neg_proj_g45", "neg_proj", 0.45, 1.0, device=args.device)
    r_ort, _ = run_n3_arm(core, ctx, "N3b_ortho_g45", "ortho", 0.45, 1.0, device=args.device)
    r_cap, _ = run_n3_arm(core, ctx, "N3c_rmscap_g45", "rms_cap", 0.45, 1.0, device=args.device)
    rows.extend([r_neg, r_ort, r_cap])

    # 在均衡高内容档 (g=0.50, wc=1.5) 上再测一次 N3a，检验高内容保护档是否获益
    r_hi_neg, _ = run_n3_arm(core, ctx, "N3a_neg_proj_g50_wc15", "neg_proj", 0.50, 1.5, device=args.device)
    rows.append(r_hi_neg)

    cost = benchmark_cost(core, ctx, device=args.device)
    save_json(OUT / "n3_conflict_and_cost.json", {
        "updated_utc": utc(),
        "diag_g45_wc10": diag_g45,
        "arms": rows,
        "cost": cost,
    })
    print("\n=== N3 诊断 & 消融完成 ===", flush=True)
    print("diag:", diag_g45, flush=True)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
