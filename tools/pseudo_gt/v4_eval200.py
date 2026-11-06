"""V4 一级评估（200 跨风格）：读 pgt_render 渲染出的 PNG，算 ΔT/ΔS/ΔM/DINO-C/LPIPS。

基线严格按 V4 协议：VAE(x)（同一批源图、同一 512px、posterior mean 编码 + TRT 解码），
CSD 原型与预处理与训练/审计完全一致；不使用 AdaIN（渲染时已强制 endpoint_adain_scale=0）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import csd_model  # type: ignore  # noqa: E402

from tools.eval_fast import (  # noqa: E402
    _load_eval_image_tensor, build_protocol,
)
from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, pgt_root  # noqa: E402
from tools.pseudo_gt.matching import encode_dino  # noqa: E402


def _to_t(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1


def _boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="pgt_render 的 cell 名（输出子目录）")
    ap.add_argument("--render-root", default="")
    ap.add_argument("--test-dir", default=str(REPO_ROOT / "data" / "test"))
    ap.add_argument("--max-src", type=int, default=10, help="每风格源图数（10 → 50 源 / 200 跨风格）")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.render_root) if args.render_root else Path(pgt_root()) / "eval_render"
    img_dir = root / args.cell / "images"
    test_images, requests, style_meta = build_protocol(
        Path(args.test_dir), max_src_samples=args.max_src, ref_pool="first",
    )
    metric = Q.Metrics()
    protos = Q.build_prototypes(metric, Q.load_latents(), set())
    proto_mat = np.stack([protos[s] for s in STYLES])

    rows = []
    for src_style_id, (src_style, paths) in sorted(test_images.items()):
        for path in paths:
            src_img = Image.open(path).convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
            z = metric.encode(_load_eval_image_tensor(path, args.image_size).unsqueeze(0).to("cuda"))
            base_img = metric.decode(z)[0]
            cs_b = csd_model.feats_pils(metric.csd, [base_img], batch=1)[0]
            cls_b, _ = encode_dino(metric.dino, [base_img], batch=1)
            src_t = _to_t(src_img)
            for tgt_style_id in sorted(style_meta):
                gen_path = img_dir / f"{src_style}_{path.stem}_to_{style_meta[tgt_style_id]['name']}.png"
                if not gen_path.exists():
                    continue
                gen_img = Image.open(gen_path).convert("RGB")
                cs_g = csd_model.feats_pils(metric.csd, [gen_img], batch=1)[0]
                cls_g, _ = encode_dino(metric.dino, [gen_img], batch=1)
                with torch.inference_mode():
                    lp = float(metric.lpips(_to_t(gen_img).unsqueeze(0).cuda(), src_t.unsqueeze(0).cuda()).item())
                rows.append({
                    "src_style": src_style, "src": path.name, "tgt_style": style_meta[tgt_style_id]["name"],
                    "cross": tgt_style_id != src_style_id,
                    "dT": float(cs_g @ proto_mat[tgt_style_id]) - float(cs_b @ proto_mat[tgt_style_id]),
                    "dS": float(cs_g @ proto_mat[src_style_id]) - float(cs_b @ proto_mat[src_style_id]),
                    "dino_c": float(cls_g[0] @ cls_b[0]),
                    "lpips": lp,
                })

    cross = [r for r in rows if r["cross"]]
    same = [r for r in rows if not r["cross"]]
    dT = np.array([r["dT"] for r in cross]);
    dS = np.array([r["dS"] for r in cross])
    dM = dT - dS
    out = {
        "cell": args.cell, "n_cross": len(cross), "n_same": len(same),
        "dT": float(dT.mean()), "dT_ci": _boot_ci(dT),
        "dS": float(dS.mean()), "dS_ci": _boot_ci(dS),
        "dM": float(dM.mean()), "dM_ci": _boot_ci(dM),
        "dino_c": float(np.mean([r["dino_c"] for r in cross])),
        "lpips": float(np.mean([r["lpips"] for r in cross])),
        "per_target": {
            t: {"dT": float(np.mean([r["dT"] for r in cross if r["tgt_style"] == t])),
                "dS": float(np.mean([r["dS"] for r in cross if r["tgt_style"] == t]),
                            ),
                "dM": float(np.mean([r["dT"] - r["dS"] for r in cross if r["tgt_style"] == t])),
                "n": sum(1 for r in cross if r["tgt_style"] == t)}
            for t in STYLES
        },
        "same_style_dT": float(np.mean([r["dT"] for r in same])) if same else float("nan"),
    }
    dest = Path(args.out) if args.out else root / args.cell / "v4_eval200.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / args.cell / "v4_eval200_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
