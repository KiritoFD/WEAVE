r"""V26-L 验收: SG-25 OOD × 任意 L1/L3 One-Shot 臂 (3060 专责 — SG 资产在本机).

docs/1120 §8 L6 验收清单 (1): SG-25 OOD. C2nt 权重已于 K 阶段测过
(sg25_c2nt_t45.json, delta_dino_s=+0.2216); 本模块对最终候选臂
S0_vgg_global 与 F2_global_teacher 重复同一 125-pair 协议.

复用 exp.v25.r3_oneshot_minimal.sg25_and_swap (零训练, 只读): swap_dist +
125 对 (dino_s 正确参考 vs 参考更换) → delta_dino_s. ckpt 需含
fm/adapter/fib_mean/fib_std (train_l1_arm 落盘格式).

用法 (3060 仓库根):  python -m exp.v26.gpu3060.sg25_v26l
"""
from __future__ import annotations

import argparse
import json

import torch

from exp.v26 import OUT, ROOT, sanitize_cuda_env, save_json, utc
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.submode_patchswd import load_convnext_tiny_offline


def run_one(pool: str, tag: str, ckpt_p, device: str = "cuda") -> dict:
    out_p = OUT / "gpu3060" / f"sg25_{tag}.json"
    if out_p.exists():
        print(f"[SG25 {tag}] Reusing {out_p}", flush=True)
        return json.loads(out_p.read_text(encoding="utf-8"))
    from exp.v25.r3_oneshot_minimal import sg25_and_swap

    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    cnx = load_convnext_tiny_offline(device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)

    swap_dist, rows = sg25_and_swap(bundle, c21, cnx, ckpt_p,
                                    [(0.45, 2.2, "t45")], device=device)
    r0 = rows[0]
    out = {"task": f"V26-L SG-25 OOD audit for {tag} (t45)",
           "updated_utc": utc(), "ckpt": str(ckpt_p),
           "swap_dist": float(swap_dist), "delta_dino_s": r0.get("delta_dino_s"),
           "n_pairs": r0.get("n_pairs"), "rows": rows}
    save_json(out_p, out)
    print(f"[SG25 {tag}] swap_dist={swap_dist:.4f} delta_dino_s={out['delta_dino_s']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tags", default="S0_vgg_global,F2_global_teacher")
    args = ap.parse_args()
    sanitize_cuda_env()
    ck_dir = ROOT / "runs/v26/oneshot_val"
    for tag in args.tags.split(","):
        run_one(args.pool, tag.strip(), ck_dir / f"{tag.strip()}.pt", device=args.device)


if __name__ == "__main__":
    main()
