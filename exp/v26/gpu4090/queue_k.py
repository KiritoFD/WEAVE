r"""V26 gpu4090 队列: K0 → SR-CFG(SR0 复用 + SR1–SR7 + 自参考审计) → K1 → K2 (串行, 幂等).

调整版分工 (docs/1120/06_4090_diagnostic_queue.md + 用户 2026-11-20 C2nt 降级决策):
  - 4090 负责零训练 C2nt 诊断 (SR-CFG / RefAttn 干预 / 题材偏置) 与 Q32 审计;
  - SG-25 OOD 因 StyleGallery 资产仅在 3060, 移至 3060 队列;
  - 每步结束打印 PROGRESS 行; 单步失败不中断队列 (记录后继续).
"""
from __future__ import annotations

import traceback

from exp.v26 import ROOT, OUT, sanitize_cuda_env, seed_all, save_json, utc


def _step(name: str, fn) -> None:
    print(f"\n[QUEUE-4090] ===== {name} start {utc()} =====", flush=True)
    try:
        fn()
        print(f"[QUEUE-4090] ===== {name} DONE {utc()} =====", flush=True)
    except Exception:
        print(f"[QUEUE-4090] ===== {name} FAILED =====\n{traceback.format_exc()}", flush=True)


def main():
    from exp.v26.diagnose_q32 import run_k0
    from exp.v26.sr_cfg import SR_ARMS, eval_sr_arm, load_sr_env, selfref_identity_audit
    from exp.v26 import run_measure_v26, arm_done, V25_RAW_C2NT_T45, V25_MEASURED_C2NT_T45
    from exp.v26.diagnose_refattn import run_k1
    from exp.v26.k2_style_subject_audit import run_k2

    sanitize_cuda_env()
    seed_all(20261185)
    pool = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu4090").mkdir(parents=True, exist_ok=True)

    # SR0 复用 4090 上的 V25 raw (同 seed 同协议)
    import shutil
    if not arm_done("SR0_t45") and V25_RAW_C2NT_T45.exists() and \
            not (OUT / "raw" / "SR0_t45" / "meta.json").exists():
        shutil.copytree(V25_RAW_C2NT_T45, OUT / "raw" / "SR0_t45")
        if V25_MEASURED_C2NT_T45.exists():
            (OUT / "measured" / "SR0_t45").mkdir(parents=True, exist_ok=True)
            shutil.copy(V25_MEASURED_C2NT_T45, OUT / "measured" / "SR0_t45" / "summary_v24.json")
            shutil.copy(V25_MEASURED_C2NT_T45, OUT / "measured" / "SR0_t45" / "summary_v26.json")
        print("[QUEUE-4090] SR0 reused V25 raw", flush=True)

    _step("K0_q32_audit", lambda: run_k0(pool, device="cuda"))

    def _sr_all():
        env = load_sr_env(pool, device="cuda")
        for arm in SR_ARMS:
            eval_sr_arm(env, arm, device="cuda")
        selfref_identity_audit(env, device="cuda")
    _step("SR_CFG_matrix", _sr_all)

    _step("K1_refattn_causal", lambda: run_k1(pool, device="cuda"))
    _step("K2_style_subject_audit", lambda: run_k2(pool, device="cuda"))
    print("\n[QUEUE-4090] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
