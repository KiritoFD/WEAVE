#!/usr/bin/env python3
"""Independent-reference-pool style evaluation for V4 outputs.

The paper protocol scores DINO-S against a single reference image (the first
image of the target style inside ``data/test``).  That reference is part of the
test set, so the score is not fully independent.  This script recomputes a
style score against a reference pool that the model never saw and that is not
part of ``data/test``:

  reference pool = author-held-out validation split latents
                   (``runs/v4/prior/indices.pt`` -> ``val_idx``),
                   30 decoded images per style -> mean feature -> prototype.

Metrics reported per generated image:
  dino_style_pool  cosine(gen DINOv2 CLS, target-style prototype)
  clip_style_pool  cosine(gen CLIP image emb, target-style prototype)
  dino_content_pool cosine(gen DINOv2 CLS, source image DINOv2 CLS)

Usage:
  python3 tools/12h/v4_refpool_eval.py <eval_out_dir> [--n-per-style 30]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(p))

from utils.compute_dino_metrics import load_dino  # noqa: E402
from utils.inference import decode_latent, load_vae  # noqa: E402
from utils.run_evaluation import (  # noqa: E402
    _clip_image_size_from_runtime,
    _clip_mean_std_from_runtime,
    _extract_clip_embeddings,
    _load_clip_from_source,
    _prepare_clip_pixels,
)
from v4.data import STYLE_NAMES, load_bank  # noqa: E402

CACHE = "/mnt/i/Latent_Style/eval_cache/hf"
HUB_CACHE = CACHE + "/hub"  # from_pretrained(cache_dir=...) wants the hub root, not HF_HOME
DINO_REPO = "facebook/dinov2-small"
CLIP_REPO = "openai/clip-vit-base-patch32"


def load_pngs(paths: list[Path], size: int = 512) -> torch.Tensor:
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BICUBIC)
            arr = np.asarray(im, dtype="uint8").copy()
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0))
    return torch.stack(imgs, 0)


@torch.inference_mode()
def dino_cls(images_01: torch.Tensor, model, device: str, batch: int = 16) -> torch.Tensor:
    """Same preprocessing as DINO_TRANSFORM, applied to 0-1 tensors in memory."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, images_01.shape[0], batch):
        chunk = images_01[i:i + batch].to(device=device, dtype=torch.float32)
        px = F.interpolate(chunk, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        px = (px - mean) / std
        o = model(px)
        out.append(F.normalize(o.last_hidden_state[:, 0, :].float(), dim=-1))
    return torch.cat(out, 0)


@torch.inference_mode()
def clip_img(images_01: torch.Tensor, model, proc, device: str,
             size: int, mean, std, batch: int = 16) -> torch.Tensor:
    out = []
    for i in range(0, images_01.shape[0], batch):
        chunk = images_01[i:i + batch]
        px = _prepare_clip_pixels(chunk, image_size=size, mean=mean, std=std).to(device)
        emb = _extract_clip_embeddings(model.get_image_features(pixel_values=px))
        out.append(F.normalize(emb.float(), dim=-1))
    return torch.cat(out, 0)


def prototypes(feats: torch.Tensor, style_ids: list[int], num_styles: int) -> torch.Tensor:
    protos = []
    for sid in range(num_styles):
        sel = feats[[i for i, s in enumerate(style_ids) if s == sid]]
        protos.append(F.normalize(sel.mean(0), dim=-1))
    return torch.stack(protos, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-per-style", type=int, default=30)
    ap.add_argument("--indices", type=Path, default=ROOT / "runs/v4/prior/indices.pt")
    ap.add_argument("--packed-dir", default=str(ROOT / "data/train/.latent_cache/packed/packed"))
    ap.add_argument("--test-dir", default=str(ROOT / "data/test"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_styles = len(STYLE_NAMES)

    # ---- reference pool from the author-held-out validation split -----------
    idx = torch.load(args.indices, map_location="cpu", weights_only=False)
    val_idx = idx["val_idx"].tolist()
    bank = load_bank(args.packed_dir)
    sel: list[int] = []
    ref_styles: list[int] = []
    counts = [0] * num_styles
    for i in val_idx:
        sid = int(bank.style_ids[i])
        if counts[sid] < args.n_per_style:
            sel.append(i)
            ref_styles.append(sid)
            counts[sid] += 1
    print(f"[refpool] reference images: {len(sel)} (per-style {counts})")

    vae = load_vae("cuda", model_id="ema", cache_dir=CACHE)
    ref_imgs = []
    for i in range(0, len(sel), 8):
        z = bank.latents.index_select(0, torch.tensor(sel[i:i + 8])).cuda()
        ref_imgs.append(decode_latent(vae, z).float().cpu())
    ref_imgs = torch.cat(ref_imgs, 0)
    del vae
    torch.cuda.empty_cache()

    dino = load_dino(DINO_REPO, device, CACHE, allow_network=False)
    from transformers import CLIPModel, CLIPProcessor
    clip_model, clip_proc = _load_clip_from_source(
        CLIPModel, CLIPProcessor, CLIP_REPO, device, local_only=True, cache_dir=HUB_CACHE)
    clip_model.eval()
    clip_size = _clip_image_size_from_runtime(clip_model, clip_proc)
    clip_mean, clip_std = _clip_mean_std_from_runtime(clip_proc)

    ref_dino = dino_cls(ref_imgs, dino, device)
    ref_clip = clip_img(ref_imgs, clip_model, clip_proc, device, clip_size, clip_mean, clip_std)
    proto_dino = prototypes(ref_dino, ref_styles, num_styles)
    proto_clip = prototypes(ref_clip, ref_styles, num_styles)

    # ---- generated images -------------------------------------------------
    rows = list(csv.DictReader((args.out_dir / "metrics.csv").open(newline="")))
    gen_paths = [args.out_dir / r["gen_image"] for r in rows]
    src_paths = [Path(args.test_dir) / r["src_style"] / r["src_image"] for r in rows]
    gen_imgs = load_pngs(gen_paths)
    src_imgs = load_pngs(src_paths)

    gen_dino = dino_cls(gen_imgs, dino, device)
    gen_clip = clip_img(gen_imgs, clip_model, clip_proc, device, clip_size, clip_mean, clip_std)
    src_dino = dino_cls(src_imgs, dino, device)

    tgt_ids = [STYLE_NAMES.index(r["tgt_style"]) for r in rows]
    src_ids = [STYLE_NAMES.index(r["src_style"]) for r in rows]

    d_key = [float(gen_dino[i] @ proto_dino[tgt_ids[i]]) for i in range(len(rows))]
    c_key = [float(gen_clip[i] @ proto_clip[tgt_ids[i]]) for i in range(len(rows))]
    d_content = [float(gen_dino[i] @ src_dino[i]) for i in range(len(rows))]

    def summarize(vals, keep):
        vs = [v for v, k in zip(vals, keep) if k]
        return round(sum(vs) / len(vs), 5) if vs else None

    cross = [src_ids[i] != tgt_ids[i] for i in range(len(rows))]
    out = {
        "dir": str(args.out_dir),
        "tag": args.tag,
        "n": len(rows),
        "n_cross": sum(cross),
        "n_same": len(rows) - sum(cross),
        "ref_pool": {"source": "validation split (author holdout)", "per_style": counts},
        "dino_style_pool": summarize(d_key, [True] * len(rows)),
        "dino_style_pool_x": summarize(d_key, cross),
        "dino_style_pool_s": summarize(d_key, [not c for c in cross]),
        "clip_style_pool": summarize(c_key, [True] * len(rows)),
        "clip_style_pool_x": summarize(c_key, cross),
        "clip_style_pool_s": summarize(c_key, [not c for c in cross]),
        "dino_content_pool": summarize(d_content, [True] * len(rows)),
        "dino_content_pool_x": summarize(d_content, cross),
        "dino_content_pool_s": summarize(d_content, [not c for c in cross]),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (args.out_dir / "refpool.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

