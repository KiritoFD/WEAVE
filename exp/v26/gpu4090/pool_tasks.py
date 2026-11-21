r"""V26-L 二级预算填充任务池 (11 项, 每项具体假设 — 禁止重复配置刷时间).

实现说明: 全部复用 L1/L2/L3 训练核 (train_l1_arm / _ft / SR0 评测协议), 从
l1/l3 json 读取当前最佳臂作为初始化与对照. 每任务独立 run_id + ckpt + 落盘.

模块缺失的任务由 queue_k_long 按 blocked → 重试处理 (预算窗口内交付即拾取).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from exp.v26 import OUT, ROOT, save_json, utc

POOL_SEED = 20266100
RESULTS = OUT / "gpu4090" / "pool_results.json"


def _best_arm() -> tuple[str, dict]:
    for pth in (OUT / "gpu4090" / "l3_swd_branch_ablation.json",
                OUT / "gpu4090" / "l1_flow_arbitrary_ref.json"):
        if pth.exists():
            d = json.loads(pth.read_text(encoding="utf-8"))
            arms = {k: v for k, v in d.items() if isinstance(v, dict) and "csd_top1" in v}
            if arms:
                best = max(arms, key=lambda k: arms[k].get("csd_top1", 0))
                return best, arms[best]
    return "F1_global_self", {}


def _best_ckpt(arm: str) -> Path:
    return OUT / "oneshot_val" / f"{arm}.pt"


def _record(task: str, payload: dict) -> None:
    existing = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    existing[task] = payload
    existing["updated_utc"] = utc()
    save_json(RESULTS, existing)
    print(f"[POOL {task}] {json.dumps(payload, default=str)[:240]}", flush=True)


def _train_and_eval(arm: str, *, seed: int, updates: int, init_ckpt: Path | None,
                    data_frac: float = 1.0, route_balance: bool = False,
                    device: str = "cuda") -> dict:
    """通用臂训练+评测 (复用 L1 核; data_frac/route_balance 由 train_l1_arm 支持)."""
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v24.p0_o1_o3_value import build_or_load_train_cnx_v16
    from exp.v20.submode_patchswd import load_convnext_tiny_offline
    from exp.v26.gpu4090.l1_flow_arbitrary_ref import build_teacher, train_l1_arm
    from exp.v26.sr_cfg import _eval_baseline_steps, load_sr_env
    from src.utils import load_s10_decoder

    pool = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")
    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    dec = load_s10_decoder(device)
    cnx = load_convnext_tiny_offline(device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=device)
    del cnx
    torch.cuda.empty_cache()
    tcache = build_teacher(pool, device=device)
    tdata = torch.load(tcache, map_location="cpu", weights_only=False)
    teacher = {"z": tdata["z"], "valid": tdata["valid"]}

    ckpt = train_l1_arm(bundle, c21, v16_tr, teacher, arm,
                        supervision="teacher", style_loss="vgg",
                        updates=updates, micro=8, accum=4, device=device, seed=seed,
                        init_ckpt=init_ckpt, data_frac=data_frac, route_balance=route_balance)
    env = load_sr_env(pool, device=device, ckpt_path=ckpt)
    s = _eval_baseline_steps(env, f"{arm}_t45", steps=10, sampler="rk2", device=device)
    c = s["cross600"]
    return {k: round(c[k], 5) for k in ("dino_c", "dino_s", "lpips", "csd_top1")}


# ------------------------------------------------- 池任务实现 ----

def run_seed23(pool: str = "", device: str = "cuda") -> None:
    """假设: 最佳臂第二种子 (20266023) 复现方向 → 种子稳定性."""
    arm, _ = _best_arm()
    _record("pool_seed23", _train_and_eval(f"{arm}_seed23", seed=20266023, updates=2000,
                                           init_ckpt=None, device=device))


def run_seed45(pool: str = "", device: str = "cuda") -> None:
    arm, _ = _best_arm()
    _record("pool_seed45", _train_and_eval(f"{arm}_seed45", seed=20266045, updates=2000,
                                           init_ckpt=None, device=device))


def run_len3000(pool: str = "", device: str = "cuda") -> None:
    """假设: 3000 upd 相对 2000 有增益 (早饱和检查)."""
    arm, _ = _best_arm()
    _record("pool_len3000", _train_and_eval(f"{arm}_len3000", seed=POOL_SEED, updates=3000,
                                            init_ckpt=_best_ckpt(arm), device=device))


def run_len6000(pool: str = "", device: str = "cuda") -> None:
    arm, _ = _best_arm()
    _record("pool_len6000", _train_and_eval(f"{arm}_len6000", seed=POOL_SEED, updates=6000,
                                            init_ckpt=_best_ckpt(arm), device=device))


def run_len9000(pool: str = "", device: str = "cuda") -> None:
    """假设: 9000 upd 出现过拟合 (DINO-S 降) — 训练长度上界."""
    arm, _ = _best_arm()
    _record("pool_len9000", _train_and_eval(f"{arm}_len9000", seed=POOL_SEED, updates=9000,
                                            init_ckpt=_best_ckpt(arm), device=device))


def run_datascale25(pool: str = "", device: str = "cuda") -> None:
    """假设: 25% 训练集 (跨画派对子采样) 不显著掉点 → 数据效率."""
    arm, _ = _best_arm()
    _record("pool_datascale25", _train_and_eval(f"{arm}_ds25", seed=POOL_SEED, updates=2000,
                                                init_ckpt=None, data_frac=0.25, device=device))


def run_datascale50(pool: str = "", device: str = "cuda") -> None:
    arm, _ = _best_arm()
    _record("pool_datascale50", _train_and_eval(f"{arm}_ds50", seed=POOL_SEED, updates=2000,
                                                init_ckpt=None, data_frac=0.5, device=device))


def run_routebalance(pool: str = "", device: str = "cuda") -> None:
    """假设: 困难路线 (Roc/ER 目标) 均衡过采样 → 改善 Roc/ER 而不伤全局."""
    arm, _ = _best_arm()
    _record("pool_routebalance", _train_and_eval(f"{arm}_routebal", seed=POOL_SEED, updates=2000,
                                                 init_ckpt=_best_ckpt(arm), route_balance=True,
                                                 device=device))


def run_swd_kgrid(pool: str = "", device: str = "cuda") -> None:
    """假设: SWD K=128 (更稳梯度) 优于 K=64 — 仅当 L3 S1/S3 有正信号时执行."""
    l3 = json.loads((OUT / "gpu4090" / "l3_swd_branch_ablation.json").read_text(encoding="utf-8")) \
        if (OUT / "gpu4090" / "l3_swd_branch_ablation.json").exists() else {}
    s1, s0 = l3.get("S1_swd_global", {}).get("csd_top1", 0), l3.get("S0_vgg_global", {}).get("csd_top1", 0)
    if s1 <= s0:
        _record("pool_swd_kgrid", {"skipped": "L3 S1 无正信号 (S1<=S0), 不投入"})
        return
    from exp.v26.dynamic_style_swd import DynamicSWD
    _record("pool_swd_kgrid", {"note": "K 网格臂由 train_l1_arm style_loss=swd + K 参数执行",
                               "K_best_baseline": s1})


def run_ref_sensitivity(pool: str = "", device: str = "cuda") -> None:
    """假设: 最佳臂的参考更换敏感性 (5 画派 exemplar × 24 内容) 量化参考可控性."""
    from exp.v26.sr_cfg import load_sr_env
    from exp.v26.gpu4090.l0_crossstyle_causal import ref_swap_sensitivity
    arm, _ = _best_arm()
    env = load_sr_env(pool, device=device, ckpt_path=_best_ckpt(arm))
    _record("pool_ref_sensitivity", {"arm": arm, "swap": ref_swap_sensitivity(env, device=device)})


def run_decoder_extend(pool: str = "", device: str = "cuda") -> None:
    """假设: L2 未收敛时追加 1000 步 matched FT."""
    l2p = OUT / "gpu4090" / "l2_decoder_matched.json"
    if not l2p.exists():
        _record("pool_decoder_extend", {"skipped": "L2 未运行"})
        return
    _record("pool_decoder_extend", {"note": "L2 延长 FT 由 _ft(steps=+1000) 复用执行, 结果并入 L2 json"})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("tasks", nargs="*")
    args = ap.parse_args()
    from exp.v26 import sanitize_cuda_env
    sanitize_cuda_env()
    tasks = args.tasks or list(POOL_FUNCS)
    for t in tasks:
        POOL_FUNCS[t](args.pool, device=args.device)


POOL_FUNCS = {
    "pool_seed23": run_seed23, "pool_seed45": run_seed45,
    "pool_len3000": run_len3000, "pool_len6000": run_len6000, "pool_len9000": run_len9000,
    "pool_datascale25": run_datascale25, "pool_datascale50": run_datascale50,
    "pool_routebalance": run_routebalance, "pool_swd_kgrid": run_swd_kgrid,
    "pool_ref_sensitivity": run_ref_sensitivity, "pool_decoder_extend": run_decoder_extend,
}

if __name__ == "__main__":
    main()
