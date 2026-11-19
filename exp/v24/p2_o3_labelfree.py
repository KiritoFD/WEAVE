"""V24 P2-P3 (3060): 真参考条件(零画派标签)伪配对训练 O3_rgb_cnx_gated (O3_label_free) + test750 & SG-25 OOD 对比。

对比 V24-P0 的 O3_rgb_cnx_gated (当时用 V23 带 5类重心先验的 1000 对伪配对):
本实验先用 O2_ref_teacher 在 3060 本地零标签生产 800 对真参考外观转移配对 (100% 不读 sid_r / bary),
然后用与 O3 完全相同的网络与训练步数训练 O3_label_free, 检验去掉五分类先验偏置后的真实参考迁移能力！

用法 (3060):
  python -m exp.v24.p2_o3_labelfree --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import torch

from . import OUT, ROOT, sanitize_cuda_env, save_json, utc
from .p0_o1_o3_value import build_or_load_train_cnx_v16, eval_o_arm, train_o_arm
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from src.utils import load_s10_decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-cand", type=int, default=960)
    ap.add_argument("--n-keep", type=int, default=800)
    ap.add_argument("--updates", type=int, default=1200)
    args = ap.parse_args()

    sanitize_cuda_env()
    teach_p = OUT / "cache/ref_teacher_pairs_3060.pt"
    if not teach_p.exists():
        # 直接在 3060 本地无标签生成 800 对 (约 15 分钟, zero file transfer!)
        from . import o2_ref_teacher_pairs as o2t
        orig_out = o2t.OUT
        sys.argv = [
            "o2_ref_teacher_pairs", "--pool", args.pool, "--device", args.device,
            "--n-cand", str(args.n_cand), "--n-keep", str(args.n_keep), "--steps", "18",
        ]
        # o2_ref_teacher_pairs 默认写入 OUT / "cache/ref_teacher_pairs_2000.pt"
        target_p = OUT / "cache/ref_teacher_pairs_2000.pt"
        if not target_p.exists():
            o2t.main()
        teach_p = target_p

    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    cross_pack = torch.load(teach_p, map_location="cpu", weights_only=False)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=args.device)

    # 暂时复用 O3_rgb_cnx_gated 规格在单独输出路径 O3_label_free 训练与评测
    from . import p0_o1_o3_value as o_mod
    o_mod.O_SPECS["O3_label_free"] = dict(o_mod.O_SPECS["O3_rgb_cnx_gated"])
    ckpt_p = train_o_arm(bundle, c21, cross_pack, v16_tr, "O3_label_free", updates=args.updates, device=args.device)
    res = eval_o_arm(bundle, c21, cnx, ckpt_p, "O3_label_free", device=args.device)

    save_json(OUT / "p2_o3_labelfree.json", {"updated_utc": utc(), "results": res})
    print("\n=== V24 O3_label_free (零画派标签真参考教师) 汇总 ===", flush=True)
    for r in res:
        print(f"{r['arm']:24s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
