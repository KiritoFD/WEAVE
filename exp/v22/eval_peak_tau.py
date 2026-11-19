"""V22 Peak ckpt × τ 窗口消融：solver tau_min/tau_hi 复核（旧 ckpt 上定的 0.28/0.66）。

用法 (4090):
  python -m exp.v22.eval_peak_tau --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch

from . import OUT, EVAL_SEED, sanitize_cuda_env, save_json, utc
from .core import PrismCore
from .eval_tmatrix import eval_t_ckpt
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from .t4 import build_or_load_t4_latents_750

PEAK = OUT / "tmatrix" / "slf_T4Ssnap1500.pt"
TAU_VARIANTS = (
    ("g42hi55", dict(tau_hi=0.55)),
)


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

    core = PrismCore(bundle, bank, PEAK, device=args.device)
    from .core import GateConfig
    results = []
    for tag, over in TAU_VARIANTS:
        res = eval_t_ckpt(core, PEAK, f"T4P{tag}", ctx, device=args.device,
                          wc_list=((1.0, "wc10"),),
                          gate=GateConfig(mode="fixed", fixed_g=0.42),
                          solver_over=over)
        results.extend(res)

    save_json(OUT / "peak_tau_sweep.json", {"updated_utc": utc(), "results": results})
    print("\n=== Peak1500 × τ 窗口（g=0.45, wc=1.0；基线 tauHi66 已测 76.83%） ===", flush=True)
    for r in results:
        if not r.get("skipped"):
            print(f"{r['arm']}: Top1={r['csd_top1']*100:.2f}% DINO-S={r['dino_s']:.4f} "
                  f"DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
