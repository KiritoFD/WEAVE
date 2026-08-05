"""TAESD encode D5-512 training set -> packed latent cache.

TAESD (madebyollin/taesd) is a tiny autoencoder for SD1.5 with:
  - scaling_factor = 1.0  (vs SD1.5 VAE's 0.18215)
  - same latent shape: 4×64×64 for 512×512 images
  - ~10x faster encode/decode than SD1.5 VAE

This script mirrors fast_vae_encode_d5.py but uses TAESD instead of SD1.5 VAE.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

D5_ROOT_REMOTE = Path(r"I:\Github\Latent_Style\Dataset\distinct5_512\train_512clean")
D5_ROOT_LOCAL = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\train_512clean")
D5_ROOT = D5_ROOT_REMOTE if D5_ROOT_REMOTE.is_dir() else D5_ROOT_LOCAL
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SIZE = 512
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TAESD_SCALE = 1.0  # TAESD scaling_factor (vs SD1.5's 0.18215)


def load_taesd(device: torch.device) -> torch.nn.Module:
    """Load TAESD from local cache."""
    from diffusers import AutoencoderTiny
    taesd_path = _ROOT / "data" / "taesd"
    model = AutoencoderTiny.from_pretrained(str(taesd_path), local_files_only=True)
    model = model.to(device=device, dtype=torch.float16).eval()
    model.requires_grad_(False)
    return model


def encode_images(taesd, image_paths: list[Path], batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode all images with TAESD, return stacked latents [N, 4, 64, 64]."""
    import cv2

    latents_all = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        images = []
        for p in batch_paths:
            buf = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (SIZE, SIZE):
                img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            images.append(img)

        # Stack: [B, H, W, C] -> [B, C, H, W]
        batch_np = np.stack(images, axis=0).astype(np.float32) / 255.0
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(
            device=device, dtype=torch.float16
        ) * 2.0 - 1.0  # normalize to [-1, 1]

        with torch.inference_mode():
            latent = taesd.encode(batch_tensor).latents
            # TAESD scale factor is 1.0, so no extra scaling needed
            latents_all.append(latent.float().cpu())

        if (i // batch_size) % 10 == 0:
            print(f"  encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    return torch.cat(latents_all, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--output_dir", type=str, default=str(_ROOT / "data" / "train_d5_taesd"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {torch.cuda.get_device_name(0)}  total {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"image_root: {D5_ROOT}")

    print("\nLoading TAESD...")
    t0 = time.perf_counter()
    taesd = load_taesd(device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    out_root = Path(args.output_dir)
    packed = out_root / "packed"
    packed.mkdir(parents=True, exist_ok=True)

    total = 0
    t_all = time.perf_counter()

    for style_id, style in enumerate(STYLES):
        style_dir = D5_ROOT / style
        paths = sorted(p for p in style_dir.iterdir() if p.suffix.lower() in EXTS)
        print(f"\nEncoding {style} ({len(paths)} images)...")
        t_style = time.perf_counter()
        latents = encode_images(taesd, paths, args.batch_size, device)
        elapsed = time.perf_counter() - t_style

        payload = {
            "schema": 1,
            "subdir": style,
            "count": len(paths),
            "files": [p.name for p in paths],
            "latents": latents,
        }
        torch.save(payload, packed / f"{style_id:02d}_{style}.pt")
        total += len(paths)
        print(f"  {style}: {len(paths)} imgs  {1000 * elapsed / len(paths):.1f} ms/img  {tuple(latents.shape)}")

    wall = time.perf_counter() - t_all
    meta = {
        "vae": "taesd",
        "scaling_factor": TAESD_SCALE,
        "image_root": str(D5_ROOT),
        "styles": STYLES,
        "total_samples": total,
        "image_size": SIZE,
        "batch_size": args.batch_size,
        "wall_sec": wall,
        "ms_per_img": 1000.0 * wall / max(total, 1),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"DONE  {total} images in {wall / 60:.2f} min  ({meta['ms_per_img']:.1f} ms/img)")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()