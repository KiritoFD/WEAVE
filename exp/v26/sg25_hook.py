r"""V26 SG-25 OOD 钩子 (3060): C2nt checkpoint 的 StyleGallery 25 类 125-pair 盲测.

docs/1120 调整版: C2nt 降级后仍欠的独立 OOD 验证 — SG-25 严格使用 StyleGallery 的
25 类 / 125 对 (源图 × OOD 参考风格), 检验参考响应是否外推到训练五画派之外.

复用 exp.v25.r3_oneshot_minimal.sg25_and_swap (零训练, 只读); 输出落 runs/v26/gpu3060/.
仅可在装有 StyleGallery 资产的 3060 上运行 (4090 无 SG root).

用法:  python -m exp.v26.sg25_hook --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import json

import torch

from . import C2NT_CKPT, OUT, ROOT, V21_CACHE, sanitize_cuda_env, save_json, utc
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.submode_patchswd import load_convnext_tiny_offline


def run_sg25(pool: str, device: str = "cuda") -> dict:
    from exp.v25.r3_oneshot_minimal import sg25_and_swap

    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(V21_CACHE, map_location="cpu", weights_only=False)
    cnx = load_convnext_tiny_offline(device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)

    swap_dist, rows = sg25_and_swap(bundle, c21, cnx, C2NT_CKPT,
                                    [(0.45, 2.2, "t45")], device=device)
    out = {
        "task": "V26 SG-25 OOD audit for C2nt_sw6_lc0.8 (t45)",
        "updated_utc": utc(),
        "ckpt": str(C2NT_CKPT).replace("\\", "/").split("runs/")[-1],
        "swap_dist": float(swap_dist),
        "rows": rows,
    }
    save_json(OUT / "gpu3060" / "sg25_c2nt_t45.json", out)
    print(f"[V26 SG25] swap_dist={swap_dist} rows={len(rows)}", flush=True)
    for r in rows:
        print(f"[V26 SG25] {json.dumps(r, default=str)[:220]}", flush=True)
    return out


def main():
    from . import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_sg25(args.pool, device=args.device)


if __name__ == "__main__":
    main()
