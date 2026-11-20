r"""V26 K2-C 正对照: 真实画派图像的裁判天花板检查 (本地 4070 可跑).

docs/1120/08_v26_long_plan.md §五-2: "全题材 Rococo Top-1=0%" 只证明失败广泛,
不能区分 (a) 模型画不出 Rococo (b) CSD 对某些内容不敏感. 本模块用**真实画作**
(holdout 300 张, 训练从未见过) 做正对照:
  - CSD 5-way: 真图被 CSD 判为其自身画派的比例 → 裁判对真风格的天花板;
  - CLIP 零样本: 真图被 CLIP 判为其自身画派的比例 → 第二独立评价视角;
  - 若 CSD 对真 Rococo 的天花板也低 → 模型 0% 部分来自裁判; 若天花板高
    (如 ≥85%) → 模型 0% 是真实迁移失败.

用法:  python -m exp.v26.k2c_positive_control --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import OUT, ROOT, sanitize_cuda_env, save_json, seed_all, utc
from exp.v19.fiber_duality import build_or_load_v19_foundation
from src.utils import decode_s10, load_s10_decoder
from src.utils.features import FeatureStack

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
N_PER_STYLE = 60  # holdout 300 = 60/画派, 全部使用


def run_positive_control(pool: str, device: str = "cuda") -> dict:
    seed_all(20261185)
    bundle = build_or_load_v19_foundation(pool, device=device)
    ho = bundle["v18_cache"]["holdout"]
    z_all, sid_all = ho["z"], ho["sid"].long()
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)

    from PIL import Image as PILImage

    def feats_of(imgs):
        pil = [PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
               for x in imgs]
        return stack.extract(pil, batch=16)

    refs = stack.csd_prototypes  # (5, 768) np.ndarray
    import numpy as np
    proto = torch.as_tensor(np.stack([refs[s] for s in STYLES])).float().to(device)

    rows = []
    import torch.nn.functional as NF
    proto = NF.normalize(proto.float(), dim=-1)
    cp = refs2_clip(device)
    for s in range(5):
        m = (sid_all == s)
        zs = z_all[m].to(device).float()
        imgs = torch.cat([decode_s10(dec, zs[i:i + 16], batch=16).clamp(0.0, 1.0)
                          for i in range(0, len(zs), 16)])
        f = feats_of(imgs)
        csd = torch.as_tensor(f["csd"]).float().to(device)
        csd = NF.normalize(csd, dim=-1)
        pred_csd = (csd @ proto.T).argmax(dim=1).cpu()
        csd_top1 = float((pred_csd == s).float().mean())
        # CLIP 零样本 (5 画派文本原型) — 第二独立评价视角
        cl = torch.as_tensor(f["clip"]).float().to(device)
        pred_clip = (cl @ cp.T).argmax(dim=1).cpu()
        clip_top1 = float((pred_clip == s).float().mean())
        rows.append({"style": STYLES[s], "n": int(m.sum()),
                     "csd_top1_true": round(csd_top1, 4),
                     "clip_top1_true": round(clip_top1, 4)})
        print(f"[K2C-PC] {STYLES[s]:18s} n={int(m.sum())} CSD={csd_top1:.3f} CLIP={clip_top1:.3f}", flush=True)

    out = {"task": "V26-K2C positive control: referee ceiling on TRUE style images (holdout 300)",
           "updated_utc": utc(), "rows": rows,
           "note": "csd_top1_true = 裁判对真实画派图像的天花板; Rococo 天花板高则模型 0% 为真实失败"}
    save_json(OUT / "gpu4090" / "k2c_positive_control.json", out)
    return out


def refs2_clip(device: str) -> torch.Tensor:
    """CLIP 文本原型: 5 画派提示词 → (5,512)."""
    from transformers import CLIPModel, CLIPProcessor
    import paths
    cache = str(paths.eval_cache_root("hf") / "hub")
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True,
                                  cache_dir=cache).to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True,
                                         cache_dir=cache)
    prompts = [f"a painting in the {s.replace('_', ' ')} style" for s in STYLES]
    ti = proc(text=prompts, return_tensors="pt", padding=True).to(device)
    import torch.nn.functional as NF
    with torch.no_grad():
        t = m.get_text_features(**ti)
        t = t if torch.is_tensor(t) else t.pooler_output
        return NF.normalize(t.float(), dim=-1)


def main():
    from . import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_positive_control(args.pool, device=args.device)


if __name__ == "__main__":
    main()
