"""Pre-encode P2A-256 training images into packed latent cache for AdaCUTLatentDataset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.legacy_utils.inference import encode_image, load_vae

P2A_IMAGE_ROOT = Path(r"I:\Github\Latent_Style\style_data\train")
P2A_STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
IMAGE_SIZE = 256


def load_image_tensor(path: Path, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(device, dtype=dtype)


def scan_style_images(style: str) -> list[Path]:
    style_dir = P2A_IMAGE_ROOT / style
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def preencode_style(
    vae: torch.nn.Module,
    style: str,
    style_id: int,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> int:
    image_paths = scan_style_images(style)
    if not image_paths:
        raise RuntimeError(f"No images found for style={style} under {P2A_IMAGE_ROOT}")

    latents = []
    print(f"  Encoding style={style} (n={len(image_paths)})...")
    t0 = time.time()
    for i in range(0, len(image_paths), batch_size):
        chunk_paths = image_paths[i : i + batch_size]
        batch = torch.cat([load_image_tensor(p, device) for p in chunk_paths], dim=0)
        with torch.no_grad():
            z = encode_image(vae, batch, device=device)
        latents.append(z.detach().cpu().float())
        if (i // batch_size) % 10 == 0 or i + batch_size >= len(image_paths):
            print(f"    {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
    print(f"  Encoded in {time.time() - t0:.1f}s")

    stack = torch.cat(latents, dim=0)
    packed_dir = output_dir / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)
    out_path = packed_dir / f"{style_id:02d}_{style}.pt"
    payload = {
        "schema": 1,
        "subdir": style,
        "count": len(image_paths),
        "files": [str(p.relative_to(P2A_IMAGE_ROOT)) for p in image_paths],
        "latents": stack,
    }
    torch.save(payload, out_path)
    print(f"  Saved {out_path} shape={tuple(stack.shape)}")
    return len(image_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(_PROJECT_ROOT / "data" / "train_p2a" / ".latent_cache"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    cache_dir = str(_PROJECT_ROOT / "eval_cache" / "hf")

    print(f"Loading SD1.5 EMA VAE on {device}...")
    vae = load_vae(device=device, model_id="ema", cache_dir=cache_dir, enable_xformers=False)
    vae.eval()
    vae.requires_grad_(False)

    total = 0
    for style_id, style in enumerate(P2A_STYLES):
        print("=" * 60)
        n = preencode_style(vae, style, style_id, output_dir, device, args.batch_size)
        total += n
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta = {
        "vae": "sd15_ema",
        "image_root": str(P2A_IMAGE_ROOT),
        "styles": P2A_STYLES,
        "total_samples": total,
        "image_size": IMAGE_SIZE,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("=" * 60)
    print(f"DONE: {total} samples encoded -> {output_dir / 'packed'}")


if __name__ == "__main__":
    main()