"""V22-P2 T4 × wc 细扫：T4(layout 固定 1.0) ckpt 在 wc∈{1.0,1.15,1.2,1.3,1.5} 五档 + 已有 wc11/wc14 → Pareto 甜点定位。

用法 (4090):
  python -m exp.v22.eval_t4_sweep --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch

from . import OUT, EVAL_SEED, config_hash, read_json, sanitize_cuda_env, save_json, utc
from .eval_tmatrix import BASE_CKPT, eval_t_ckpt
from .core import PrismCore
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from .t4 import build_or_load_t4_latents_750

T4_CKPT = OUT / "tmatrix" / "slf_T4.pt"
SWEEP_WCS = ((1.0, "wc100"), (1.15, "wc115"), (1.2, "wc120"), (1.3, "wc130"), (1.5, "wc150"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(OUT.parent.parent.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=str(T4_CKPT))
    ap.add_argument("--tag", default="T4s")
    ap.add_argument("--wcs", default="1.0,1.15,1.2,1.3,1.5")
    args = ap.parse_args()

    sanitize_cuda_env()

    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    ctx = {
        "tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750, "d16_750": d16_750,
        "eps_750": eps_750, "z_t4": z_t4, "dec": None,  # dec 在 eval_t_ckpt 内不用；实际 ctx 传 dec 供 decode
    }
    from src.utils import load_s10_decoder
    ctx["dec"] = load_s10_decoder(args.device)

    wcs = tuple((float(w), f"wc{w.replace('.', '')}") for w in args.wcs.split(","))
    core = PrismCore(bundle, bank, Path(args.ckpt), device=args.device)
    res = eval_t_ckpt(core, Path(args.ckpt), args.tag, ctx, device=args.device, wc_list=wcs)
    save_json(OUT / f"{args.tag}_wc_sweep.json", {"updated_utc": utc(), "results": res})

    print(f"\n=== {args.tag} × wc 细扫（fixed gate） ===", flush=True)
    for r in res:
        if r.get("skipped"):
            print(f"{r['arm']}: skipped", flush=True)
        else:
            print(f"{r['arm']}: Top1={r['csd_top1']*100:.2f}% DINO-S={r['dino_s']:.4f} "
                  f"DINO-C={r['dino_c']:.4f} Aff={r['style_affinity']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
