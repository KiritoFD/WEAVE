"""V22 T4S 快照曲线评测：slf_T4Ssnap{300..3600}.pt × wc=1.0 固定门控 → 训练步数-Top1 曲线峰值定位。

用法 (4090):
  python -m exp.v22.eval_t4_snaps --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch

from . import OUT, EVAL_SEED, read_json, sanitize_cuda_env, save_json, utc
from .eval_tmatrix import eval_t_ckpt
from .core import PrismCore
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from .t4 import build_or_load_t4_latents_750

SNAPS = (600, 900, 1200, 1500, 2100, 2700, 3300, 3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(OUT.parent.parent.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wc", type=float, default=1.0)
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
    for s in SNAPS:
        ckpt = OUT / "tmatrix" / f"slf_T4Ssnap{s}.pt"
        if not ckpt.exists():
            print(f"[SNAP] SKIP snap{s} (missing)", flush=True)
            continue
        core = PrismCore(bundle, bank, ckpt, device=args.device)
        res = eval_t_ckpt(core, ckpt, f"T4Ss{s}", ctx, device=args.device,
                          wc_list=((args.wc, f"wc{args.wc}"),))
        results.extend(res)
        del core
        torch.cuda.empty_cache()

    save_json(OUT / "t4s_snap_curve.json", {"updated_utc": utc(), "wc": args.wc, "results": results})
    print(f"\n=== T4S 快照曲线（wc={args.wc}, fixed gate） ===", flush=True)
    for r in results:
        if not r.get("skipped"):
            print(f"updates={r['arm'].replace('T4Ss', ''):>5s}: Top1={r['csd_top1']*100:.2f}% "
                  f"DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
