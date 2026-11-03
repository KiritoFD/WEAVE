"""Extract frozen DINOv2 patch-token caches for Route C.

Usage:
    python3 tools/extract_dino_cache.py --src data/train --out /mnt/i/Latent_Style/dino_cache/content_5k \
        --model_dir /mnt/i/Latent_Style/weights/dinov2-base --res 224 --batch 64
    python3 tools/extract_dino_cache.py --src data/test --out /mnt/i/Latent_Style/dino_cache/refs_test \
        --model_dir /mnt/i/Latent_Style/weights/dinov2-base --res 224 --batch 64

Output layout:
    <out>/<style>/<image_stem>.pt   -> {"tokens": float16 [N,768], "grid": [h,w], ...}
    <out>/manifest.json             -> model/layer/res/dim/counts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def build_transform(res: int):
    return transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(args.device).eval()
    patch = int(getattr(model.config, "patch_size", 14))
    grid = args.res // patch
    tf = build_transform(args.res)

    items = []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            items.append(p)
    counts, done = {}, 0
    for i in range(0, len(items), args.batch):
        chunk = items[i:i + args.batch]
        imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(args.device)
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(pixel_values=imgs, interpolate_pos_encoding=True)
        toks = o.last_hidden_state[:, 1:, :].float().half().cpu()  # drop CLS
        for p, t in zip(chunk, toks):
            rel = p.relative_to(src)
            d = out / rel.parent
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"tokens": t.clone(), "grid": [grid, grid], "res": args.res,
                        "model": "dinov2-base", "layer": "last_hidden_state"}, d / (p.stem + ".pt"))
            counts[str(rel.parent)] = counts.get(str(rel.parent), 0) + 1
            done += 1
        if done % 512 == 0 or done == len(items):
            print(f"extracted {done}/{len(items)}", flush=True)

    manifest = {
        "model": "facebook/dinov2-base", "layer": "last_hidden_state",
        "resolution": args.res, "patch_size": patch, "grid": [grid, grid],
        "token_dim": int(model.config.hidden_size), "num_tokens": grid * grid,
        "num_samples": done, "per_style": counts, "dtype": "float16",
        "source": str(src),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("MANIFEST", json.dumps(manifest))


if __name__ == "__main__":
    main()
