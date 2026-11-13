"""VAE encode R5 training set into packed latent cache for WEAVE training.

Reads: I:\\datasets\\r5\\train\\<Style>\\*.jpg (5000 images)
Writes: I:\\Github\\Latent_Style\\WEAVE\\data\\train_r5\\.latent_cache\\packed\\<id>_<Style>.pt

Each output .pt = {"schema":1, "subdir":Style, "count":N, "files":[...], "latents":Tensor[N,4,64,64]}
"""
import sys
import time
from pathlib import Path

# Add WEAVE root to path (remote path)
WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
sys.path.insert(0, str(WEAVE_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn

from tools.legacy_utils.inference import load_vae

# Config
R5_TRAIN_ROOT = Path(r"I:\datasets\r5\train")
OUTPUT_ROOT = WEAVE_ROOT / "data" / "train_r5" / ".latent_cache" / "packed"
STYLES = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]
SIZE = 512
SCALE = 0.18215
BATCH_SIZE = 8  # safe for RTX 3060 12GB
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16
CACHE_DIR = str(WEAVE_ROOT / "eval_cache" / "hf")


class EncoderOnlyVAE(nn.Module):
    def __init__(self, encoder, quant_conv, scaling_factor):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean * self.scaling_factor


def build_encoder():
    vae = load_vae(
        device=str(DEVICE), model_id="ema", cache_dir=CACHE_DIR,
        enable_xformers=False, compile_decoder=False, compile_dtype=DTYPE,
    ).to(device=DEVICE, dtype=DTYPE).eval().requires_grad_(False)
    scale = float(getattr(vae.config, "scaling_factor", SCALE))
    enc = EncoderOnlyVAE(vae.encoder, vae.quant_conv, scale).to(
        device=DEVICE, dtype=DTYPE, memory_format=torch.channels_last,
    ).eval().requires_grad_(False)
    del vae.decoder, vae.post_quant_conv, vae
    torch.cuda.empty_cache()
    return enc


def decode_bgr(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def load_image_chw(path):
    img = decode_bgr(path)
    if img is None:
        raise RuntimeError(f"failed to decode {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


@torch.inference_mode()
def encode_style(enc, style_dir):
    """Encode all jpgs in style_dir, return (files, latents_tensor)."""
    paths = sorted(p for p in style_dir.iterdir() if p.suffix.lower() == ".jpg")
    print(f"  {style_dir.name}: {len(paths)} images")

    all_latents = []
    t0 = time.perf_counter()

    for i in range(0, len(paths), BATCH_SIZE):
        chunk = paths[i:i + BATCH_SIZE]
        arrays = [load_image_chw(p) for p in chunk]
        # Pad last batch if needed
        while len(arrays) < BATCH_SIZE:
            arrays.append(arrays[-1])

        batch_uint8 = np.stack(arrays, axis=0)
        x = torch.from_numpy(batch_uint8).to(DEVICE, dtype=DTYPE)
        x = x.mul_(1.0 / 127.5).sub_(1.0).contiguous(memory_format=torch.channels_last)
        z = enc(x)

        n_real = len(chunk)
        all_latents.append(z[:n_real].detach().float().cpu())

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"    batch {i // BATCH_SIZE + 1}/{(len(paths) + BATCH_SIZE - 1) // BATCH_SIZE}")

    elapsed = time.perf_counter() - t0
    stack = torch.cat(all_latents, 0)
    files = [p.name for p in paths]
    print(f"  done in {elapsed:.1f}s ({1000 * elapsed / len(paths):.1f} ms/img), shape={tuple(stack.shape)}")
    return files, stack


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"R5 train root: {R5_TRAIN_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Batch size: {BATCH_SIZE}")

    # Verify train data
    for style in STYLES:
        d = R5_TRAIN_ROOT / style
        if not d.exists():
            print(f"ERROR: {d} not found")
            return
        n = len(list(d.glob("*.jpg")))
        print(f"  {style}: {n} jpgs")

    # Build encoder
    print("\nBuilding encoder-only VAE...")
    enc = build_encoder()
    print("Encoder ready")

    # Encode each style
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    t_all = time.perf_counter()

    for style_id, style in enumerate(STYLES):
        style_dir = R5_TRAIN_ROOT / style
        files, latents = encode_style(enc, style_dir)

        out_path = OUTPUT_ROOT / f"{style_id:02d}_{style}.pt"
        payload = {
            "schema": 1,
            "subdir": style,
            "count": len(files),
            "files": files,
            "latents": latents,
        }
        torch.save(payload, out_path)
        total += len(files)
        print(f"  Saved: {out_path.name} ({len(files)} samples)")

    wall = time.perf_counter() - t_all
    print(f"\n{'=' * 60}")
    print(f"DONE: {total} images in {wall / 60:.2f} min")
    print(f"Output: {OUTPUT_ROOT}")

    # Also write manifest.json
    manifest_path = OUTPUT_ROOT.parent / "manifest.json"
    import json
    manifest = {
        "schema": 1,
        "data_root": str(R5_TRAIN_ROOT),
        "style_subdirs": STYLES,
        "styles": {},
    }
    for style in STYLES:
        d = R5_TRAIN_ROOT / style
        files = sorted(p.name for p in d.glob("*.jpg"))
        manifest["styles"][style] = {"count": len(files), "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
