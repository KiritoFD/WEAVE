"""V22-P1 消融队列：复现核验 → A(锚点) → B(CFG) → C(门控) → D(原型) → CF(反事实) → Same150 拆分。

用法 (4090):
  python -m exp.v22.run --queue --pool ../pseudo_gt_24h/orig_mean_fp16
  python -m exp.v22.run --repro-only --pool ../pseudo_gt_24h/orig_mean_fp16   # 仅 P0 复现核验
  python -m exp.v22.run --groups A,B --pool ...                                # 子集
"""
from __future__ import annotations

import argparse
import time

import torch

from . import (
    EVAL_SEED,
    OUT,
    ROOT,
    RawWriterV22,
    config_hash,
    export_raw_pngs,
    file_sha,
    git_sha,
    read_json,
    run_measure_v22,
    sanitize_cuda_env,
    save_json,
    utc,
    validate_raw_meta,
)
from .core import D2_SOLVER, D5_SOLVER, D6_SOLVER, CFGConfig, GateConfig, PrismCore, latent_change_rate
from .t4 import T4_HASH, build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import apply_latent_chroma_shift, build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from src.utils.protocol import pick_wrong_cross_tid

CKPT = ROOT / "runs/v20/slf_flow/slf_core_v2.pt"


# ---------------- arm 规格定义 ----------------

def build_arms() -> dict[str, dict]:
    """每组只改变一个组件; alias 臂不重复生成（引用既有官方 measured）。"""
    arms: dict[str, dict] = {}

    def add(name, *, solver=None, gate=None, cfg=None, style_mode=None, anchor="t4",
            same_mode="bypass", wrong_tgt=False, alias=None, measure=True, repro_of=None,
            pullback_beta=0.0):
        arms[name] = {
            "name": name, "solver": dict(solver or D5_SOLVER),
            "gate": gate or {}, "cfg": cfg or {},
            "style_mode": style_mode, "anchor": anchor,
            "same_mode": same_mode, "wrong_tgt": wrong_tgt,
            "alias": alias, "measure": measure, "repro_of": repro_of,
            "pullback_beta": pullback_beta,
        }

    # ---- P0 复现核验（不跑 measure，只逐任务比对 z_final）----
    add("repro_D5", solver=D5_SOLVER, repro_of="runs/v20/raw/V20_D5_core_slf_c64", measure=False)
    add("repro_D6", solver=D6_SOLVER, repro_of="runs/v20/raw/V20_D6_core_slf_c69", measure=False)
    add("repro_D2", solver=D2_SOLVER, repro_of="runs/v20/raw/V20_D2_core_slf_bal", measure=False)

    # ---- A组: 锚点贡献（base D5, gate original, branches full）----
    add("A0_pure_noise", anchor=None)
    add("A1_src_anchor", anchor="src")
    add("A2_prism1_anchor", anchor="prism1")
    # A3 = D5 本身（T4 锚点）；A4 = T4-only → 复用 v19 官方 measured，不重生成
    add("A4_t4_only", anchor="t4", alias="V19_T4_sweet_vgg_nosid")

    # ---- B组: CFG 分支（固定 D5 锚点/起点/步数）----
    add("B0_sc_only", cfg={"branches": "sc_only"})
    add("B1_cs_j", cfg={"branches": "cs_j"})
    add("B2_no_wp", cfg={"branches": "no_wp"})
    add("B3_no_wc", cfg={"branches": "no_wc"})
    add("B4_no_rescale", cfg={"rescale_phi": 0.0})
    # B5 = D5 原版 → alias

    # ---- C组: 门控解耦（branches full, base D5）----
    add("C0_fixed_median", gate={"mode": "fixed", "fixed_g": 0.5})
    add("C1_tau_only", gate={"mode": "original", "drive_layout": False, "drive_guidance": False})
    add("C2_layout_only", gate={"mode": "original", "drive_tau": False, "drive_guidance": False})
    add("C3_guidance_only", gate={"mode": "original", "drive_tau": False, "drive_layout": False})
    add("C5_quantile", gate={"mode": "quantile"})
    # C4 = D5 原版 → alias

    # ---- D组: 原型（固定门控 fixed_g=0.5 解耦 原型→dsd→gate 耦合）----
    dg = {"mode": "fixed", "fixed_g": 0.5}
    add("D_K1_barycenter", gate=dg, style_mode="core1")
    add("D_K4_merged", gate=dg, style_mode="core4")
    add("D_K8_random", gate=dg, style_mode="random")
    add("D_K8_core_fixedgate", gate=dg, style_mode="core8")
    # D_K8_core_originalgate = D5 → alias

    # ---- CF组: 反事实锚点对照（Fix 3）----
    add("CF2_wrong_tgt_same_t4", wrong_tgt=True, anchor="t4_keep")
    add("CF4_wrong_tgt_chroma_anchor", wrong_tgt=True, anchor="chroma_wrong")
    # CF1 = D5 (correct tgt + T4 anchor); CF3 = A1 (correct tgt + src anchor)

    # ---- Same150 拆分（Fix 2）----
    add("S_blind_same150", same_mode="blind")
    add("S_self_ref_same150", same_mode="self")

    # ---- W组: wc 连续旋钮扫描（B3 发现 wc 压低 Top-1；固定中位数门控 × wc 扫描）----
    wg = {"mode": "fixed", "fixed_g": 0.5}
    w_solver = dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66)  # tau 由 fixed g 决定
    add("W_wc00_fixedgate", gate=dg, solver=dict(w_solver, wc_fig=0.0, wc_abs=0.0))
    add("W_wc04_fixedgate", gate=dg, solver=dict(w_solver, wc_fig=0.4, wc_abs=0.4))
    add("W_wc09_fixedgate", gate=dg, solver=dict(w_solver, wc_fig=0.9, wc_abs=0.9))
    add("W_wc14_fixedgate", gate=dg, solver=dict(w_solver, wc_fig=1.4, wc_abs=1.4))

    # ---- W2组: 组合优化（W 组揭示 wc 旋钮后的下一轮）----
    add("X1_prism1_wc00", anchor="prism1", gate={"mode": "fixed", "fixed_g": 0.5},
        solver=dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, wc_fig=0.0, wc_abs=0.0))
    add("X2_prism1_wc12", anchor="prism1", gate={"mode": "fixed", "fixed_g": 0.5},
        solver=dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, wc_fig=1.2, wc_abs=1.2))
    add("X3_wc12_fixedgate", gate=dg, solver=dict(w_solver, wc_fig=1.2, wc_abs=1.2))
    add("X4_wc11_fixedgate", gate=dg, solver=dict(w_solver, wc_fig=1.1, wc_abs=1.1))
    add("X5_wc13_fixedgate", gate=dg, solver=dict(w_solver, wc_fig=1.3, wc_abs=1.3))
    add("X6_wc00_wspboost", gate=dg, solver=dict(w_solver, wc_fig=0.0, wc_abs=0.0, wsp_fig_late=4.0))

    # ---- P2组: 双分支重训模型评测（3060 one-shot 旁路；ckpt 由 --ckpt 指定）----
    add("P2_dual_vsc", cfg={"branches": "sc_only"})
    add("P2_dual_cs_j", cfg={"branches": "cs_j"})

    # ---- N2组: P3-N2 粗结构信任域（推理期拉回粗亮度，救 wc 中档内容；无训练）----
    add("N2_wc11_b15", gate={"mode": "fixed", "fixed_g": 0.5},
        solver=dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, wc_fig=1.1, wc_abs=1.1),
        pullback_beta=0.15)
    add("N2_wc11_b30", gate={"mode": "fixed", "fixed_g": 0.5},
        solver=dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, wc_fig=1.1, wc_abs=1.1),
        pullback_beta=0.30)
    add("N2_wc14_b15", gate={"mode": "fixed", "fixed_g": 0.5},
        solver=dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, wc_fig=1.4, wc_abs=1.4),
        pullback_beta=0.15)
    return arms


DEFAULT_GROUPS = {
    "repro": ["repro_D5", "repro_D6", "repro_D2"],
    "A": ["A0_pure_noise", "A1_src_anchor", "A2_prism1_anchor"],
    "B": ["B0_sc_only", "B1_cs_j", "B2_no_wp", "B3_no_wc", "B4_no_rescale"],
    "C": ["C0_fixed_median", "C1_tau_only", "C2_layout_only", "C3_guidance_only", "C5_quantile"],
    "D": ["D_K1_barycenter", "D_K4_merged", "D_K8_random", "D_K8_core_fixedgate"],
    "CF": ["CF2_wrong_tgt_same_t4", "CF4_wrong_tgt_chroma_anchor"],
    "S": ["S_blind_same150", "S_self_ref_same150"],
    "W": ["W_wc00_fixedgate", "W_wc04_fixedgate", "W_wc09_fixedgate", "W_wc14_fixedgate"],
    "W2": ["X1_prism1_wc00", "X2_prism1_wc12", "X3_wc12_fixedgate", "X4_wc11_fixedgate", "X5_wc13_fixedgate", "X6_wc00_wspboost"],
    "P2E": ["P2_dual_vsc", "P2_dual_cs_j"],
    "W2": ["X1_prism1_wc00", "X2_prism1_wc12", "X3_wc12_fixedgate", "X4_wc11_fixedgate", "X5_wc13_fixedgate", "X6_wc00_wspboost"],
    "P2E": ["P2_dual_vsc", "P2_dual_cs_j"],
    "N2": ["N2_wc11_b15", "N2_wc11_b30", "N2_wc14_b15"],
}


# ---------------- 生成 ----------------

@torch.no_grad()
def run_arm(core: PrismCore, spec: dict, ctx: dict, device: str = "cuda") -> dict:
    tasks = ctx["tasks"]
    z_src_750 = ctx["z_src_750"]
    y16_750, d16_750 = ctx["y16_750"], ctx["d16_750"]
    eps_750 = ctx["eps_750"]
    dec = ctx["dec"]

    name = spec["name"]
    cfg_hash = config_hash({k: v for k, v in spec.items() if k != "repro_of"})
    out_dir = OUT / "raw" / name
    if out_dir.joinpath("meta.json").exists():
        old = read_json(out_dir / "meta.json")
        if old.get("cfg_hash") == cfg_hash and validate_raw_meta(old):
            print(f"[V22] SKIP {name} (same cfg_hash, complete)", flush=True)
            return {"skipped": True, "arm": name, "cfg_hash": cfg_hash}
        out_dir = OUT / "raw" / f"{name}__{cfg_hash[:8]}"
        if out_dir.joinpath("meta.json").exists() and validate_raw_meta(read_json(out_dir / "meta.json")):
            print(f"[V22] SKIP {out_dir.name} (complete)", flush=True)
            return {"skipped": True, "arm": out_dir.name, "cfg_hash": cfg_hash}

    gate = GateConfig(**spec["gate"]) if spec["gate"] else GateConfig()
    cfg = CFGConfig(**spec["cfg"]) if spec["cfg"] else CFGConfig()
    solver = spec["solver"]

    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])

    # 锚点选择（750 级，一次性）
    if spec["anchor"] is None:
        anchor_750 = None                      # A0 纯噪声
    elif spec["anchor"] == "src":
        anchor_750 = z_src_750.clone()
    elif spec["anchor"] == "prism1":
        anchor_750 = ctx["prism1_anchor"]
    elif spec["anchor"] == "chroma_wrong":
        tid_wrong_all = pick_wrong_cross_tid(sid_all, tid_all)
        anchor_750 = apply_latent_chroma_shift(z_src_750.float(), tid_wrong_all, ctx["priors_cpu"], alpha=0.20)
    else:  # "t4" / "t4_keep"
        anchor_750 = ctx["z_t4"].clone()

    # 风格条件（750 级）
    u_override, dsd_override = None, None
    if spec["style_mode"]:
        u_override, dsd_override = core.build_style_condition(
            d16_750.to(device).float(), tid_all.to(device), mode=spec["style_mode"],
            task_index=torch.arange(750),
        )
    u_self = core.self_reference_u(ctx["fib_750"]) if spec["same_mode"] == "self" else None

    wr = RawWriterV22(out_dir.name, dec=dec, device=device)
    t0 = time.time()
    rate_cross, rate_same = [], []
    stats_acc = {"g_mean": [], "tau_mean": [], "lscale_mean": []}
    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        grp = tasks[sl]
        B = len(grp)
        sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
        tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
        same = sid_p == tid_t
        tid_u = pick_wrong_cross_tid(sid_p, tid_t) if spec["wrong_tgt"] else tid_t

        if u_override is not None:
            u_b = u_override[sl].to(device)
            dsd_b = dsd_override[sl].to(device)
        elif spec["same_mode"] == "self":
            u_core8, dsd_core8 = core.build_style_condition(d16_750[sl].to(device).float(), tid_u, mode="core8")
            u_self_b = u_self[sl].to(device).float()
            u_b = torch.where(same.view(B, 1), u_self_b, u_core8)
            dsd_b = dsd_core8
        else:
            u_b, dsd_b = core.build_style_condition(d16_750[sl].to(device).float(), tid_u, mode="core8")

        anchor_b = None if anchor_750 is None else anchor_750[sl].to(device).float()

        z_gen, stats = core.integrate_flow(
            eps_750[sl], anchor_b, y16_750[sl], d16_750[sl], tid_u,
            u_style=u_b, dsd_ref=dsd_b, gate=gate, cfg=cfg, solver=solver,
        )
        for k in stats_acc:
            stats_acc[k].append(stats[k])

        # Fix 2: bypass（协议旁路）只作用于输出替换；blind/self 让 same 任务走模型
        if spec["same_mode"] == "bypass":
            z_out = torch.where(same.view(B, 1, 1, 1), z_src_750[sl].to(device).float(), z_gen)
        else:
            z_out = z_gen

        # P3-N2: 粗结构信任域（推理期一次修正，把粗亮度场向锚点拉回 beta）
        beta_pb = float(spec.get("pullback_beta") or 0.0)
        if beta_pb > 0.0 and anchor_b is not None:
            from .core import coarse_luminance_pullback
            z_out = coarse_luminance_pullback(z_out, anchor_b, ctx["w_luma"], ctx["b_luma"], beta=beta_pb)

        rate = latent_change_rate(z_out, z_src_750[sl])
        if bool(same.all()):
            rate_same.append(rate)
        elif not bool(same.any()):
            rate_cross.append(rate)
        wr.write_batch(z_out, grp)

    meta = wr.finalize(
        cfg_hash=cfg_hash, git_sha=git_sha(),
        ckpt_sha256=file_sha(CKPT), bank_sha256=file_sha(ctx["bank_path"]),
        tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
        t4_sha256=str(T4_HASH), t4_source=ctx["t4_source"], seed=EVAL_SEED,
        model_cfg={"base_dim": 144, "u_dim": 128},
        infer_cfg={"solver": spec["solver"], "gate": spec["gate"], "cfg_branches": spec["cfg"],
                   "style_mode": spec["style_mode"], "anchor": spec["anchor"],
                   "same_mode": spec["same_mode"], "wrong_tgt": spec["wrong_tgt"]},
        wall_sec=round(time.time() - t0, 2),
        g_stats={k: round(sum(v) / len(v), 4) for k, v in stats_acc.items()},
        latent_change_rate_cross=round(sum(rate_cross) / max(1, len(rate_cross)), 4),
        latent_change_rate_same=round(sum(rate_same) / max(1, len(rate_same)), 4) if rate_same else None,
    )
    print(f"[V22 Gen] {out_dir.name} | 750 in {meta['wall_sec']}s ({round(meta['wall_sec']*1000/750,1)} ms/img)", flush=True)
    return {"skipped": False, "arm": out_dir.name, "cfg_hash": cfg_hash}


# ---------------- 复现核验 ----------------

def compare_repro(core: PrismCore, spec: dict, ctx: dict) -> dict:
    from exp.v20.slf_flow import integrate_slf_core_batch

    ref_dir = ROOT / spec["repro_of"]
    meta = read_json(ref_dir / "meta.json")
    assert validate_raw_meta(meta), f"{ref_dir} 不完整"
    orig = []
    for b in meta["batches"]:
        d = torch.load(ref_dir / b["file"], map_location="cpu", weights_only=False)
        orig.append(d["z_final"].float())
    z_orig = torch.cat(orig, dim=0)

    tasks = ctx["tasks"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    model = core.model

    def _run(gen_fn) -> torch.Tensor:
        z_new = torch.zeros(750, 4, 64, 64)
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            z_gen, _ = gen_fn(sl)
            z_new[sl] = torch.where(same_all[sl].view(-1, 1, 1, 1), ctx["z_src_750"][sl].float(), z_gen.cpu())
        return z_new

    solver_kw = {k: v for k, v in spec["solver"].items()}
    rescale = solver_kw.pop("rescale_phi", 0.48)

    # 三方比对: ① V20 参考实现同进程重跑（自一致性基线 / 实现等价性）
    #           ② V22 求解器  ③ 各自与原始 raw（跨进程 GPU 非确定性底线）的差距
    def _v20_arm(sl):
        z = integrate_slf_core_batch(
            model, ctx["z_t4"][sl].to(core.device), ctx["eps_750"][sl].to(core.device),
            ctx["y16_750"][sl].to(core.device).float(), ctx["d16_750"][sl].to(core.device).float(),
            tid_all[sl].to(core.device), core.bank, rescale_phi=rescale, **solver_kw,
        )
        return z.cpu(), None

    def _v22_arm(sl):
        z, st = core.integrate_flow(
            ctx["eps_750"][sl], ctx["z_t4"][sl], ctx["y16_750"][sl], ctx["d16_750"][sl],
            tid_all[sl].to(core.device), solver=dict(spec["solver"]),
        )
        return z.cpu(), st

    z_v20 = _run(_v20_arm)
    z_v22 = _run(_v22_arm)

    def _diff(a, b):
        d = (a - b).abs()
        per_img = d.amax(dim=(1, 2, 3))
        return {
            "max": round(float(d.max()), 6), "mean": round(float(d.mean()), 8),
            "n_within_1e4": int((per_img < 1e-4).sum()), "n_within_1e3": int((per_img < 1e-3).sum()),
            "p999": round(float(per_img.quantile(0.999)), 6),
        }

    rep = {
        "arm": spec["name"], "repro_of": spec["repro_of"],
        "v22_vs_v20_rerun": _diff(z_v22, z_v20),
        "v20_rerun_vs_orig": _diff(z_v20, z_orig),
        "v22_vs_orig": _diff(z_v22, z_orig),
        "impl_bitwise_equal": bool((z_v22 - z_v20).abs().max() == 0),
        "gpu_nondet_floor_max": _diff(z_v20, z_orig)["max"],
    }
    print(f"[V22 REPRO] {spec['name']} vs {spec['repro_of']}: "
          f"impl_equal={rep['impl_bitwise_equal']} v22_vs_v20={rep['v22_vs_v20_rerun']['max']:.2e} "
          f"v20_vs_orig={rep['gpu_nondet_floor_max']:.2e} v22_vs_orig={rep['v22_vs_orig']['max']:.2e} "
          f"(cross-only; Same150 bypass 逐位一致)", flush=True)
    return rep


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repro-only", action="store_true")
    ap.add_argument("--groups", default="repro,A,B,C,D,CF,S")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--ckpt", default=str(CKPT), help="主模型 checkpoint（3060 上指向 prism_dual_branch.pt）")
    args = ap.parse_args()

    sanitize_cuda_env()
    from src.utils import load_s10_decoder

    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    core = PrismCore(bundle, bank, args.ckpt, device=args.device)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, t4_src = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    fib_750 = bundle["fnd"]["test750_fib"].float()
    g_eval = torch.Generator().manual_seed(EVAL_SEED)
    eps_750 = torch.randn(z_src_750.shape, generator=g_eval)
    dec = load_s10_decoder(args.device)
    priors_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}

    # A2 锚点: PRISM-1 (V19 S2) 一次性推理
    print("[V22] Computing PRISM-1 anchor (A2) once...", flush=True)
    from exp.v19.routed_student import SpatiallyRoutedChordNet

    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(args.device).eval()
    ck = torch.load(ROOT / "runs/v19/students/s2_prism1_v2.pt", map_location="cpu", weights_only=False)
    stu.load_state_dict(ck["model"])
    fnd = bundle["fnd"]
    priors = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    fiber_enc = core._ensure_fiber()
    u_src_all = fiber_enc.encode(fib_750.to(args.device).float())
    bary = fiber_enc.class_barycenters()
    P_Y = priors["P_Y"]
    c64_750 = bundle["test_struct"]["c_64"].float()
    sid_all_t = torch.tensor([t["source_style_id"] for t in tasks], device=args.device)
    tid_all_t = torch.tensor([t["target_style_id"] for t in tasks], device=args.device)
    same_t = sid_all_t == tid_all_t
    z0_750 = torch.load(ROOT / "runs/v9_local_4070/cache/n2c_z.pt", map_location="cpu", weights_only=False)["z0"].float()
    prism1_anchor = torch.zeros(750, 4, 64, 64)
    with torch.no_grad():
        for st in range(0, 750, 50):
            sl = slice(st, min(st + 50, 750))
            zs = z_src_750[sl].to(args.device).float()
            z0_b = z0_750[sl].to(args.device)
            u_tgt = bary[tid_all_t[sl]]
            z_anc = apply_latent_chroma_shift(z0_b, tid_all_t[sl], priors, alpha=0.20)
            chord = u_tgt - u_src_all[sl]
            z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
            delta = stu(z_anc, z_luma, c64_750[sl].to(args.device).float(), d16_750[sl].to(args.device).float(), chord, same_mask=same_t[sl])
            z_out = z_anc + delta
            z_out[same_t[sl]] = zs[same_t[sl]]
            prism1_anchor[sl] = z_out.detach().cpu().float()
    del stu
    torch.cuda.empty_cache()

    ctx = {
        "tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750, "d16_750": d16_750,
        "fib_750": fib_750, "eps_750": eps_750, "z_t4": z_t4, "dec": dec,
        "priors_cpu": priors_cpu, "prism1_anchor": prism1_anchor,
        "bank_path": ROOT / "runs/v20/cache/core_knn_bank.pt",
        "t4_source": t4_src,
        "w_luma": bundle["v18_cache"]["luma_proj"]["w"],
        "b_luma": bundle["v18_cache"]["luma_proj"]["b"],
    }

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    arm_names: list[str] = []
    for g in groups:
        arm_names.extend(DEFAULT_GROUPS.get(g, [g]))
    specs = build_arms()

    results = {
        "updated_utc": utc(), "git_sha": git_sha(), "t4_source": t4_src, "t4_hash": str(T4_HASH),
        "ckpt_sha256": file_sha(CKPT), "arms": {}, "repro": [],
    }

    # 1) P0 复现核验
    repro_names = [n for n in arm_names if specs[n]["repro_of"]]
    if repro_names:
        for n in repro_names:
            results["repro"].append(compare_repro(core, specs[n], ctx))
        save_json(OUT / "p0_repro_report.json", results["repro"])
        ok = all(r["impl_bitwise_equal"] and r["v22_vs_orig"]["max"] <= r["gpu_nondet_floor_max"] * 1.5
                 for r in results["repro"])
        print(f"[V22 P0] Repro verdict: {'PASS' if ok else 'FAIL'} "
              f"(判据: V22 与 V20 同进程逐位等价, 且跨进程偏差 ≤ GPU 非确定性底线×1.5)", flush=True)
        if args.repro_only:
            save_json(OUT / "queue_state.json", results)
            return

    # 2) 生成 + 评测队列
    for name in arm_names:
        spec = specs[name]
        if spec["repro_of"] or spec["alias"]:
            continue
        res = run_arm(core, spec, ctx, device=args.device)
        results["arms"][name] = res
        save_json(OUT / "queue_state.json", results)
        if res.get("skipped"):
            continue
        if spec["measure"]:
            run_measure_v22(res["arm"], device=args.device)
        if not args.no_png:
            export_raw_pngs(OUT / f"raw/{res['arm']}", OUT / f"png_750/{res['arm']}", max_workers=8)

    save_json(OUT / "queue_state.json", results)
    print("[V22] Queue complete.", flush=True)


if __name__ == "__main__":
    main()
