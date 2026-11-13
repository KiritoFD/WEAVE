#!/usr/bin/env python3
"""Guarded, minimal 12GB compatibility patch for official StyleGallery.

Run INSIDE a fresh clone of https://github.com/iiiiiiiword/StyleGallery.
Does not modify the style algorithm. Changes: optional depth import, usable
no-depth switch, fp32 PCA, fp32 tiny matching transformer, gradient checkpointing.
No 12GB hardware result is implied: a CUDA smoke test is mandatory.
"""
import argparse
import os
from pathlib import Path


def replace_once(src: str, old: str, new: str, name: str) -> str:
    if new in src:
        print(f"[already patched] {name}")
        return src
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: expected 1 occurrence, got {n}; check pinned repo commit")
    print(f"[patch] {name}")
    return src.replace(old, new, 1)


def patch(root: Path):
    p = root / "pipeline.py"
    t = p.read_text(encoding="utf8")
    t = replace_once(t,
        "from pretrained_models.dpv2.depth_anything_v2.dpt import DepthAnythingV2",
        "try:\n    from pretrained_models.dpv2.depth_anything_v2.dpt import DepthAnythingV2\nexcept ImportError:\n    DepthAnythingV2 = None  # optional when --use_depth is disabled",
        "optional Depth Anything V2 import")
    t = replace_once(t,
        "features_2d = features.squeeze(0).permute(1, 2, 0).reshape(-1, features.shape[1]).to(device)",
        "features_2d = features.squeeze(0).permute(1, 2, 0).reshape(-1, features.shape[1]).to(device=device, dtype=torch.float32)",
        "PCA in fp32")
    t = replace_once(t,
        "latent_feature = latent_feature.squeeze(0).permute(1, 2, 0).reshape(-1, 4).to(device)",
        "latent_feature = latent_feature.squeeze(0).permute(1, 2, 0).reshape(-1, 4).to(device=device, dtype=torch.float32)",
        "latent feature fp32 for clustering")
    t = replace_once(t,
        "        ).to(content_features.device)\n\n        dinov2_model =",
        "        ).to(device=content_features.device, dtype=torch.float32)\n\n        dinov2_model =",
        "semantic matching transformer in fp32")
    # diffusers>=0.40 compat: enable_vae_slicing moved from pipeline to VAE
    if "self.enable_vae_slicing()" in t:
        t = t.replace("        self.enable_vae_slicing()",
                      "        self.vae.enable_slicing()  # diffusers>=0.40 compat", 1)
        print("[patch] vae slicing moved to VAE (diffusers>=0.40)")
    t = replace_once(t,
        "attended_features = transformer(valid_features.t()).t()",
        "attended_features = transformer(valid_features.t().float()).t()",
        "fp32 matching transformer input")
    t = replace_once(t,
        'dinov2_model = AutoModel.from_pretrained("pretrained_models/facebook/dinov2-base").to(self.device)',
        'dinov2_device = "cpu" if os.environ.get("STYLEGALLERY_DINO_CPU", "0") == "1" else self.device\n        dinov2_model = AutoModel.from_pretrained("pretrained_models/facebook/dinov2-base").eval().to(dinov2_device)',
        "optional CPU DINO model")
    t = replace_once(t,
        'inputs = processor(images=image, return_tensors="pt").to(device)',
        'inputs = processor(images=image, return_tensors="pt").to(next(dinov2_model.parameters()).device)',
        "DINO input follows model device")
    t = replace_once(t,
        "        encoder = 'vitl' \n\n        model = DepthAnythingV2(**model_configs[encoder])",
        "        encoder = 'vitl' \n        if DepthAnythingV2 is None:\n            raise ImportError('Depth Anything V2 source is missing; run with --no-use_depth or install it separately')\n\n        model = DepthAnythingV2(**model_configs[encoder])",
        "explicit depth guard")
    p.write_text(t, encoding="utf8")

    d = root / "demo_accelerate.py"
    t = d.read_text(encoding="utf8")
    t = replace_once(t,
        'parser.add_argument("--use_depth", action="store_true", default=True)',
        'parser.add_argument("--use_depth", action=argparse.BooleanOptionalAction, default=False)',
        "disable depth by default; --use_depth to opt in")
    t = replace_once(t,
        "        c_ratio=c_ratio\n    )",
        "        c_ratio=c_ratio,\n        enable_gradient_checkpoint=True\n    )",
        "UNet checkpointing for latent optimization")
    d.write_text(t, encoding="utf8")

    for fp in [p, d]:
        compile(fp.read_text(encoding="utf8"), str(fp), "exec")
    print("OK: patched files compile. Next: 12GB CUDA smoke test, not guaranteed yet.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="StyleGallery clone root")
    args = ap.parse_args()
    patch(Path(args.repo).resolve())
