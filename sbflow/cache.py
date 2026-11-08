"""Freeze DCAE image latents and pixel thumbnails used for the mini-batch OT cost.

Accepts MULTIPLE image roots so one style cache can be the union of several pools
(e.g. `wikiart_new_512/<Style>` + `distinct5_512/train/<Style>`), which is the
canonical WEAVE dataset defined in `data.md`.

Usage:
  python -m sbflow.cache --style Impressionism \
      --images datasets/wikiart_new_512/Impressionism datasets/distinct5_512/train/Impressionism \
      --out runs/blog_sbflow/cache5/Impressionism.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def load_dcae(model_name, device):
    from diffusers import AutoencoderDC

    ae = AutoencoderDC.from_pretrained(model_name).eval().to(device)
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def image_tensor(p, size):
    with Image.open(p) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        t = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    return t


def gather(roots, dedupe=True):
    """Union of image files under several roots, optionally de-duplicated by name."""
    seen, files = set(), []
    for r in roots:
        root = Path(r)
        if not root.exists():
            raise ValueError(f"Missing image root: {root}")
        for p in sorted(x for x in root.rglob("*") if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")):
            key = p.name if dedupe else str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="one or more image roots (union)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="", help="style label stored in the cache")
    ap.add_argument("--ae", default="mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dedupe", action="store_true", default=True, help="drop repeated file names across roots")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    ap.add_argument("--include-dino", action="store_true",
                    help="also cache DINOv2 CLS features for semantic OT")
    ap.add_argument("--dino-model", default="facebook/dinov2-small")
    args = ap.parse_args()

    files = gather(args.images, args.dedupe)
    if not files:
        raise ValueError(f"No images under {args.images}")
    print(f"[cache] {len(files)} images from {len(args.images)} root(s)", flush=True)

    ae = load_dcae(args.ae, args.device)
    latent_list, thumb_list, dino_list = [], [], []
    if args.include_dino:
        from transformers import AutoImageProcessor, AutoModel

        dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)
        dino_model = AutoModel.from_pretrained(args.dino_model).to(args.device).eval()
        for q in dino_model.parameters():
            q.requires_grad_(False)

    with torch.inference_mode():
        for start in range(0, len(files), args.batch):
            batch = torch.stack([image_tensor(p, args.image_size) for p in files[start:start + args.batch]])
            thumb = (F.interpolate(batch, size=(32, 32), mode="area") * 255).round().clamp(0, 255).byte()
            x = (batch * 2 - 1).to(args.device)
            if args.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latent = ae.encode(x).latent
            else:
                latent = ae.encode(x).latent
            # Diffusers AutoencoderDC `.encode(...).latent`, NOT `.latent_dist.mean`.
            sf = float(ae.config.scaling_factor)
            latent_list.append((latent.float() * sf).half().cpu())
            thumb_list.append(thumb.cpu())
            if args.include_dino:
                pils = [Image.open(p).convert("RGB") for p in files[start:start + args.batch]]
                inp = dino_processor(images=pils, return_tensors="pt").to(args.device)
                feats = dino_model(**inp).last_hidden_state[:, 0].float()
                dino_list.append(F.normalize(feats, dim=-1).half().cpu())
                for im in pils:
                    im.close()
            if start % (args.batch * 32) == 0:
                print(f"[cache] {start}/{len(files)}", flush=True)

    out = {"latent": torch.cat(latent_list), "thumb": torch.cat(thumb_list),
           "ids": [str(p.resolve()) for p in files], "ae": args.ae, "style": args.style,
           "roots": list(args.images),
           "scaling_factor": sf, "image_size": args.image_size,
           "pixel_input_range": "-1..1", "thumbnail_range": "0..255"}
    if args.include_dino:
        out["dino"] = torch.cat(dino_list)
        out["dino_model"] = args.dino_model
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print("saved", path, tuple(out["latent"].shape), flush=True)


if __name__ == "__main__":
    main()
