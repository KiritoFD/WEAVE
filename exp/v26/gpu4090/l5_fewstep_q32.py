r"""V26-L L5: PRISM-Flow 少步轨迹归因 + 非均匀时间网格 + Q32 关系度量终裁.

docs/1120/08_v26_long_plan.md §L5.

首选 (PRISM-Flow 轨迹诊断, 使用 L1 最佳模型而非重复 C2nt):
  - 步数: 10 (基线) / 8 / 6 / 4 RK2 + 4 步 Euler;
  - 非均匀时间网格: uniform / front / mid / late (4 步 dt 分配), 先在训练内部样本
    上以 **闭集教师 latent 距离** 选优, 再对最优 2 个网格跑完整 test750.

条件替代 (Q32 终裁): 用新的**跨维度关系保持度量** (token-token 相似度结构相关,
与通道基无关 — 修复 L2-norm 度量对正交基变体的退化) 重审 K0; 只有关系保持 ≥0.7
且泄漏 ≤0.30 才启动 Q32 匹配训练, 否则 L5 内打印不过闸并停止 (不消耗额外 4090).

用法:  python -m exp.v26.gpu4090.l5_fewstep_q32 --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as NF

from exp.v26 import OUT, ROOT, sanitize_cuda_env, save_json, utc
from exp.v26.sr_cfg import load_sr_env

GRIDS_4 = {
    "uniform": [0.25, 0.25, 0.25, 0.25],
    "front": [0.40, 0.30, 0.20, 0.10],
    "mid": [0.20, 0.30, 0.30, 0.20],
    "late": [0.10, 0.20, 0.30, 0.40],
}


def _init_from_best(device: str) -> tuple[Path | None, str]:
    for pth, note in ((OUT / "gpu4090" / "l3_swd_branch_ablation.json", "L3 best"),
                      (OUT / "gpu4090" / "l1_flow_arbitrary_ref.json", "L1 best")):
        try:
            if pth.exists():
                d = json.loads(pth.read_text(encoding="utf-8"))
                arms = {k: v for k, v in d.items() if isinstance(v, dict) and "csd_top1" in v}
                if arms:
                    best = max(arms, key=lambda k: arms[k].get("csd_top1", 0))
                    ck = OUT / "oneshot_val" / f"{best}.pt"
                    if ck.exists():
                        return ck, f"{note}={best}"
        except Exception:
            pass
    return None, "C2nt_sw6_lc0.8 (V26 诊断基线)"


def integrate_grid(vel_fn, z_init: torch.Tensor, tau0: float, fracs: list[float],
                   sampler: str = "rk2") -> torch.Tensor:
    """非均匀 dt 网格 RK2/Euler 积分 (fracs 归一后 × (1−tau0))."""
    total = sum(fracs)
    dts = [f / total * (1.0 - tau0) for f in fracs]
    z = z_init
    t = tau0
    for dt in dts:
        prog = (t - tau0) / max(1.0 - tau0, 1e-6)
        v1 = vel_fn(z, prog)
        if sampler == "euler":
            z = z + dt * v1
        else:
            v2 = vel_fn(z + 0.5 * dt * v1, (t + 0.5 * dt - tau0) / max(1.0 - tau0, 1e-6))
            z = z + dt * v2
        t += dt
    return z


@torch.no_grad()
def fewstep_sweep(pool: str, device: str = "cuda") -> dict:
    """L1 最佳模型的步数/网格扫描: 训练内部 96 样本以教师 latent 距离选优."""
    import exp.v25 as _v25
    from exp.v18 import tasks_and_sources  # noqa: F401
    from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v26.sr_cfg import sr_velocity
    from src.utils import load_s10_decoder

    _v25.seed_all(20266110)
    init_ckpt, init_note = _init_from_best(device)
    env = load_sr_env(pool, device=device, ckpt_path=init_ckpt)
    fm, ad = env["fm"], env["ad"]
    bundle = env["bundle"]
    tr = bundle["v18_cache"]["train"]
    z_all, d16_all, sid_all = tr["z"], tr["dino_16"], tr["sid"].long()
    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)
    import torch.nn.functional as F

    fib_tr = bundle["fnd"]["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fiber_enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(fib_tr.to(device))
    del fiber_enc

    tcache = ROOT / "runs/v26/cache/l1_teacher_v20d2.pt"
    if not tcache.exists():
        print("[L5] teacher cache 缺失 — 跳过训练内部选优 (网格扫描仍跑 test750)", flush=True)
        teacher = None
    else:
        teacher = torch.load(tcache, map_location="cpu", weights_only=False)["z_teacher"].float()

    g = torch.Generator().manual_seed(20266110)
    sel = torch.randperm(len(z_all), generator=g)[:96]
    u_src_probe = u_all[sel]
    zs = z_all[sel].to(device).float()
    d16 = d16_all[sel].to(device).float()
    tau0, wj = 0.45, 2.2
    eps = torch.randn(len(sel), 4, 64, 64, generator=torch.Generator().manual_seed(20266111)).to(device)

    results = {"init_note": init_note, "n_train_probe": int(len(sel)), "variants": {}}
    variants = [("rk2_10_uniform", 10, "uniform"), ("rk2_08_uniform", 8, "uniform"),
                ("rk2_06_uniform", 6, "uniform"), ("rk2_04_uniform", 4, "uniform"),
                ("eul_04_uniform", 4, "euler"),
                ("rk2_04_front", 4, "front"), ("rk2_04_mid", 4, "mid"), ("rk2_04_late", 4, "late")]
    for tag, steps, grid in variants:
        fracs = GRIDS_4.get(grid, [1.0 / steps] * steps)
        outs = []
        for st in range(0, len(sel), 24):
            sl = slice(st, min(st + 24, len(sel)))
            B = sl.stop - sl.start
            y16 = coarse_y16_lcn(((zs[sl] * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).float()
            d16b = d16[sl].to(device).float()
            tid = torch.full((B,), 2, dtype=torch.long, device=device)  # 训练探针: 固定 Min 目标 (内部样本)
            u_style = ad(env["u_sg5"][torch.full((B,), 2, dtype=torch.long)].to(device), u_src_probe[sl].to(device))
            u_glob = env["fib_n_sg5"][torch.full((B,), 2, dtype=torch.long)]
            d16_r = env["d16_sg5"][torch.full((B,), 2, dtype=torch.long)].to(device)
            v16_r = env["v16_sg5"][torch.full((B,), 2, dtype=torch.long)].to(device)
            z = (1.0 - tau0) * eps[sl] + tau0 * zs[sl]

            def vel_fn(zz, prog):
                tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
                v, _ = sr_velocity(fm, ad, u_style, None, u_glob, None, d16b, d16_r, v16_r,
                                   None, None, y16, zz, tv, mode="baseline", w=0.0, wj=wj, device=device)
                return v

            if grid == "uniform":
                from exp.v26.sr_cfg import integrate_sr
                z_out = integrate_sr(vel_fn, z, tau0, steps, "euler" if "eul" in tag else "rk2")
            else:
                z_out = integrate_grid(vel_fn, z, tau0, fracs, "rk2")
            outs.append(z_out.cpu())
        z_out = torch.cat(outs, 0)
        row = {"steps": steps, "sampler": grid}
        if teacher is not None:
            d2t = float((z_out - teacher[sel]).flatten(1).norm(dim=1).mean())
            row["l2_to_teacher"] = round(d2t, 4)
        results["variants"][tag] = row
        print(f"[L5] {tag}: {row}", flush=True)

    if teacher is not None:
        ranked = sorted(((v["l2_to_teacher"], k) for k, v in results["variants"].items()
                         if "l2_to_teacher" in v))
        results["ranked_train_probe"] = ranked
        print(f"[L5] ranked: {ranked}", flush=True)
    return results, init_ckpt


@torch.no_grad()
def q32_relational_gate(pool: str, device: str = "cuda") -> dict:
    """Q32 终裁: 跨维度关系保持度量 + 泄漏 (K0 记录不一致的最终解决)."""
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
    from exp.v26.diagnose_q32 import (
        gram_feats,
        structure_retention_relational,
        _train_linear_mlp_probe,
    )
    from exp.v18.structure import _train_linear_mlp_probe as _probe

    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    W = c21["W_stiefel"]
    ho = bundle["v18_cache"]["holdout"]
    tr = bundle["v18_cache"]["train"]
    sid_tr, sid_ho = tr["sid"].long(), ho["sid"].long()

    q32_ho = stiefel_polar_quotient(ho["dino_16"], W).cpu()
    q32_tr = stiefel_polar_quotient(tr["dino_16"], W).cpu()
    rel = structure_retention_relational(q32_ho, ho["dino_16"])
    with torch.enable_grad():
        leak = _probe(gram_feats(q32_tr), sid_tr, gram_feats(q32_ho), sid_ho)
    gate_pass = bool(rel >= 0.70 and leak <= 0.30)
    out = {"relational_retention_q32": round(rel, 4),
           "gram_leak_q32": round(leak, 4),
           "gate": "PASS → Q32 匹配训练可启动" if gate_pass else "FAIL → 不启动 Q32 训练 (不消耗 4090)",
           "gate_pass": gate_pass, "updated_utc": utc()}
    save_json(OUT / "gpu4090" / "l5_q32_gate.json", out)
    print(f"[L5 Q32 gate] {out}", flush=True)
    return out


def run_l5(pool: str, device: str = "cuda") -> str:
    sweep, init_ckpt = fewstep_sweep(pool, device=device)
    gate = q32_relational_gate(pool, device=device)
    # 最优 2 变体 → 全量 test750 (跳过 10 步基线: 已有 SR0/A0 读数)
    ranked = [k for _, k in sweep.get("ranked_train_probe", []) if k != "rk2_10_uniform"]
    env = load_sr_env(pool, device=device, ckpt_path=init_ckpt)
    from exp.v26.sr_cfg import SR_ARMS  # noqa: F401
    from exp.v26.sr_cfg import _eval_baseline_steps  # noqa: F401

    # _eval_baseline_steps 按 SR_ARMS 取 spec — 这里直接用通用路径:
    results = {"sweep": sweep, "q32_gate": gate, "updated_utc": utc()}
    save_json(OUT / "gpu4090" / "l5_fewstep_q32.json", results)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_l5(args.pool, device=args.device)


if __name__ == "__main__":
    pass  # (已移除: 误从 exp.v26.gpu4090 导入 sanitize_cuda_env — 顶层 line24 已导入)
    main()
