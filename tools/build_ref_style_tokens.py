"""Build per-style DINOv2 reference tokens for Route C evaluation.

Evaluation uses one reference image per style (``first_image_of_target_style_test_dir``).
This script picks that same image and stores its DINOv2 patch tokens, so the
evaluated model sees the reference branch it was trained with.

Tokens are read from an existing cache produced by ``tools/extract_dino_cache.py``
(recommended, no GPU needed); if the cache is missing the image is encoded on the
fly with the same preprocessing / model call.

Usage:
    python3 tools/build_ref_style_tokens.py \
        --src data/test --cache_dir /mnt/i/Latent_Style/dino_cache/refs_test \
        --out /mnt/i/Latent_Style/dino_cache/refs_test_first.pt

Output payload:
    {"style_names": [...], "tokens": float16 [num_styles, N, D],
     "model": "dinov2-base", "res": 224, "grid": [h, w], "per_style": {...}}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _first_image(style_dir: Path) -> Path | None:
    # Mirrors run_evaluation.py: sorted(*.jpg) + sorted(*.png), and the extracted
    # cache keeps the same alphabetical order for every extension.
    paths = []
    for ext in IMG_EXT:
        paths.extend(sorted(p for p in style_dir.glob(f"*{ext}")))
    return paths[0] if paths else None


def _encode_online(image_path: Path, model_dir: str, res: int, device: str) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModel

    tf = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    pixel = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        out = model(pixel_values=pixel, interpolate_pos_encoding=True)
    return out.last_hidden_state[0, 1:, :].float().cpu()  # drop CLS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="", help="tools/extract_dino_cache.py output for the same src")
    ap.add_argument("--model_dir", default="", help="local dinov2-base snapshot (needed only without cache)")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src = Path(args.src)
    style_names = sorted(d.name for d in src.iterdir() if d.is_dir())
    if not style_names:
        raise SystemExit(f"no style subdirectories under {src}")

    tokens, per_style, missing = [], {}, []
    for name in style_names:
        image = _first_image(src / name)
        if image is None:
            raise SystemExit(f"no image in {src / name}")
        cached = Path(args.cache_dir) / name / (image.stem + ".pt") if args.cache_dir else None
        if cached is not None and cached.exists():
            payload = torch.load(cached, map_location="cpu", weights_only=False)
            tok = payload["tokens"].float()
            source = "cache"
        else:
            if not args.model_dir:
                missing.append(f"{name}/{image.name}")
                continue
            tok = _encode_online(image, args.model_dir, args.res, args.device)
            source = "online"
        tokens.append(tok)
        per_style[name] = {"image": image.name, "source": source, "tokens": int(tok.shape[0])}
        print(f"[{name}] {image.name} via {source} -> {tuple(tok.shape)}", flush=True)

    if missing:
        raise SystemExit(f"missing cache entries and no --model_dir to re-encode: {missing}")

    grid = int(round(tokens[0].shape[0] ** 0.5))
    payload = {
        "style_names": style_names,
        "tokens": torch.stack(tokens).half(),
        "model": "dinov2-base",
        "layer": "last_hidden_state",
        "res": args.res,
        "grid": [grid, grid],
        "per_style": per_style,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print("WROTE", str(out), json.dumps({k: v for k, v in payload.items() if k != "tokens"}))


if __name__ == "__main__":
    main()
