r"""V26-L L-PC 前置修正 (子进程入口) — K2-A 更正重跑 (margin bug + AUC) + K2-C 真图正对照.

与 queue_k_long 子进程隔离机制配套: 原 _t_lpc 内联实现抽取为独立 main (),
保证 run_k2 全量重跑不会覆盖已入库的 k2a_corrected / k2c 正对照段落.

用法:  python -m exp.v26.gpu4090.l_pc --pool <pool>
"""
from __future__ import annotations

import argparse

from exp.v26 import ROOT, sanitize_cuda_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v26.k2_style_subject_audit import k2a_discriminability, save_k2a_results
    from exp.v26.k2c_positive_control import run_positive_control

    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    a = k2a_discriminability(bundle, device=args.device)
    save_k2a_results(a)
    run_positive_control(args.pool, device=args.device)


if __name__ == "__main__":
    main()
