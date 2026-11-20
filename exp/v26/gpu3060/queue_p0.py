r"""V26 gpu3060 队列 (调整版): P0-D 少步扫描 → SG-25 OOD → 单轮 SWD 受控试验 (串行, 幂等).

调整版预算 (docs/1120 + 用户 C2nt 降级决策): 3060 最多再给 C2nt 4–6 小时 —
  1. P0-D 零训练少步/采样器扫描 (5 臂, ~2.5h): C2nt t45 × steps {10,8,6,4} RK2 + 4 步 Euler;
  2. SG-25 OOD 盲测 (~0.7h): C2nt t45 × StyleGallery 25 类 125 对 (仅 3060 有 SG 资产);
  3. 单轮 SWD 受控试验 (~3h): S1fix_vgg (修复采样+原 VGG) vs S1fix_swd (修复采样+动态 SWD),
     各 1200 updates + t45 全量评测 — 停机条件: 两臂均无相近内容水平改善 → 冻结 C2nt 配方.
"""
from __future__ import annotations

import traceback

from exp.v26 import OUT, ROOT, sanitize_cuda_env, seed_all, utc


def _step(name: str, fn) -> None:
    print(f"\n[QUEUE-3060] ===== {name} start {utc()} =====", flush=True)
    try:
        fn()
        print(f"[QUEUE-3060] ===== {name} DONE {utc()} =====", flush=True)
    except Exception:
        print(f"[QUEUE-3060] ===== {name} FAILED =====\n{traceback.format_exc()}", flush=True)


def main():
    from exp.v26.sg25_hook import run_sg25

    sanitize_cuda_env()
    seed_all(20261185)
    pool = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")
    (OUT / "gpu3060").mkdir(parents=True, exist_ok=True)

    _step("P0D_few_step_sweep", lambda: _few_step(pool))

    _step("SG25_OOD_c2nt_t45", lambda: run_sg25(pool, device="cuda"))

    _step("SWD_trial_S1fix_vgg", lambda: _swd_arm(pool, "vgg"))
    _step("SWD_trial_S1fix_swd", lambda: _swd_arm(pool, "swd"))
    print("\n[QUEUE-3060] ALL DONE", flush=True)


def _few_step(pool: str) -> None:
    from exp.v26.sr_cfg import FEW_STEP_ARMS, eval_few_step_arm, load_sr_env

    env = load_sr_env(pool, device="cuda")
    for arm in FEW_STEP_ARMS:
        eval_few_step_arm(env, arm, device="cuda")


def _swd_arm(pool: str, style_loss: str) -> None:
    import torch
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v24.p0_o1_o3_value import build_or_load_train_cnx_v16
    from exp.v20.submode_patchswd import load_convnext_tiny_offline
    from exp.v26.gpu3060.train_c2nt_v26 import train_c2nt_v26
    from exp.v26.sr_cfg import _eval_baseline_steps, load_sr_env
    from src.utils import load_s10_decoder

    bundle = build_or_load_v19_foundation(pool, device="cuda")
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    dec = load_s10_decoder("cuda")
    cnx = load_convnext_tiny_offline("cuda").eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device="cuda")

    arm = f"V26_S1fix_{style_loss}"
    ckpt = train_c2nt_v26(bundle, c21, v16_tr, arm, style_loss=style_loss, device="cuda")
    env = load_sr_env(pool, device="cuda", ckpt_path=ckpt)
    _eval_baseline_steps(env, f"{arm}_t45", steps=10, sampler="rk2", device="cuda")


if __name__ == "__main__":
    main()
