"""构建伪GT实验的共享资产：均值 latent、DINO token、检索表。

所有路径经 ``paths.asset_root()``；不写任何绝对路径。可断点续跑。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from tools.pseudo_gt.matching import DEVICE, encode_dino, load_dino_model, assert_cuda  # noqa: E402
from tools.legacy_utils.inference import download_vae_with_fallback  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_IDS = {s: i for i, s in enumerate(STYLES)}
DINO_SOURCE = "facebook/dinov2-small"


def pgt_root() -> Path:
    return Path(paths.asset_root()) / "pseudo_gt_24h"


def work_root() -> Path:
    return pgt_root() / "work"


def _stem(path: Path) -> str:
    return path.stem


def enumerate_images() -> List[dict]:
    """合并两个来源：wikiart_new_512（train/val）+ distinct5_512/train（额外 5,000）。

    返回统一的 ``id -> {path, style, source_set, split}`` 记录。
    """
    items: List[dict] = []
    wikiart = REPO_ROOT / "datasets/wikiart_new_512"
    manifest_line = wikiart / "manifest.jsonl"
    if not manifest_line.exists():
        raise FileNotFoundError(f"缺少 {manifest_line}")
    for line in manifest_line.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        style = row["style"]
        if style not in STYLE_IDS:
            raise ValueError(f"未知风格 {style}")
        items.append(
            {
                "id": f"wikiart__{style}__{Path(row['filename']).stem}",
                "stem": Path(row["filename"]).stem,
                "path": str(wikiart / row["filename"]),
                "style": style,
                "style_id": STYLE_IDS[style],
                "source_set": "wikiart_new",
                "split": row.get("split", "train"),
            }
        )
    d5 = REPO_ROOT / "datasets/distinct5_512/train"
    for style in STYLES:
        sdir = d5 / style
        if not sdir.exists():
            raise FileNotFoundError(sdir)
        for p in sorted(sdir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            items.append(
                {
                    "id": f"d5__{style}__{p.stem}",
                    "stem": p.stem,
                    "path": str(p),
                    "style": style,
                    "style_id": STYLE_IDS[style],
                    "source_set": "distinct5",
                    "split": "train",
                }
            )
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"image id 冲突 {dup} 个（跨来源重名），需加前缀")
    return items


def save_manifest(items: List[dict]) -> Path:
    path = work_root() / "images_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(items),
        "train_count": sum(1 for it in items if it["split"] == "train"),
        "val_count": sum(1 for it in items if it["split"] == "val"),
        "by_style": {s: sum(1 for it in items if it["style"] == s) for s in STYLES},
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_manifest() -> List[dict]:
    return json.loads((work_root() / "images_manifest.json").read_text(encoding="utf-8"))["items"]


def _load_rgb(path: str, size: int = 512) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return im


def _to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0


@torch.inference_mode()
def build_mean_latents(items: List[dict], *, batch: int = 8, resume: bool = True) -> dict:
    """按风格写 ``{sid:02d}_{Style}.pt``：``latent_dist.mean * scaling``（fp16）。"""
    out_dir = pgt_root() / "orig_mean_fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，禁止 CPU 回退")
    vae = download_vae_with_fallback("stabilityai/sd-vae-ft-ema", device=DEVICE)
    for p in vae.parameters():
        if p.device.type != "cuda":
            raise RuntimeError("VAE 不在 cuda 上")
    stats: Dict[str, dict] = {}
    for style in STYLES:
        sid = STYLE_IDS[style]
        target = out_dir / f"{sid:02d}_{style}.pt"
        subset = [it for it in items if it["style"] == style]
        if target.exists() and resume:
            stats[style] = {"path": str(target), "skipped": True, "count": len(subset)}
            print(f"[skip] {target.name} 已存在（{len(subset)} 张）", flush=True)
            continue
        t0, peak, latents = time.time(), 0.0, []
        for i in range(0, len(subset), batch):
            chunk = subset[i : i + batch]
            pixels = torch.stack([_to_tensor(_load_rgb(c["path"])) for c in chunk]).to(DEVICE)
            assert_cuda(pixels, "vae input")
            pixels = pixels * 2.0 - 1.0
            pixels = pixels.to(torch.float16).contiguous(memory_format=torch.channels_last)
            dist = vae.encode(pixels).latent_dist
            latents.append((dist.mean * vae.config.scaling_factor).to(torch.float16).cpu())
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            if (i // batch) % 20 == 0:
                done = min(i + batch, len(subset))
                print(f"  {style} {done}/{len(subset)}  {time.time()-t0:.1f}s  peak={peak:.2f}GiB", flush=True)
        data = torch.cat(latents, 0)
        torch.save(
            {
                "schema": "mean_latent_fp16_v1",
                "subdir": style,
                "count": len(subset),
                "files": [c["path"] for c in subset],
                "ids": [c["id"] for c in subset],
                "latents": data,
                "scaling_factor": float(vae.config.scaling_factor),
                "vae": "stabilityai/sd-vae-ft-ema",
            },
            target,
        )
        stats[style] = {
            "path": str(target),
            "skipped": False,
            "count": len(subset),
            "seconds": round(time.time() - t0, 2),
            "sec_per_image": round((time.time() - t0) / max(len(subset), 1), 4),
            "peak_gib": round(peak, 2),
            "bytes": target.stat().st_size,
        }
        print(f"[done] {style}: {stats[style]}", flush=True)
    (work_root() / "mean_latents_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return stats


@torch.inference_mode()
def build_dino(items: List[dict], *, batch: int = 16, resume: bool = True) -> dict:
    out_dir = pgt_root() / "dino_224"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_dino_model(device=DEVICE, allow_network=False)
    stats: Dict[str, dict] = {}
    for style in STYLES:
        target = out_dir / f"{style}.pt"
        subset = [it for it in items if it["style"] == style]
        if target.exists() and resume:
            stats[style] = {"path": str(target), "skipped": True, "count": len(subset)}
            print(f"[skip] {style} dino 已存在", flush=True)
            continue
        t0, cls_all, pat_all = time.time(), [], []
        for i in range(0, len(subset), batch):
            imgs = [_load_rgb(c["path"]) for c in subset[i : i + batch]]
            cls, pat = encode_dino(model, imgs, batch=batch)
            cls_all.append(cls.cpu().half())
            pat_all.append(pat.cpu().half())
            if (i // batch) % 20 == 0:
                print(f"  dino {style} {min(i+batch, len(subset))}/{len(subset)}", flush=True)
        torch.save(
            {
                "model": DINO_SOURCE,
                "style": style,
                "count": len(subset),
                "ids": [c["id"] for c in subset],
                "cls": torch.cat(cls_all, 0),
                "patches": torch.cat(pat_all, 0),
            },
            target,
        )
        stats[style] = {"path": str(target), "count": len(subset), "seconds": round(time.time() - t0, 1)}
        print(f"[done] dino {style}: {stats[style]}", flush=True)
    (work_root() / "dino_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def build_retrieval(items: List[dict], *, topk: int = 8) -> dict:
    """训练池内、**按目标风格分别**检索 Top-8 内容候选（gt-build §4 第一级）。

    只允许 train split 作为源与供体；每源图对另外 4 个风格各存一组 top-8，
    避免把 4 个风格混在一起排序导致目标风格候选不足。
    """
    out_dir = pgt_root() / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx = {it["id"]: i for i, it in enumerate(items)}
    all_cls, all_ids = [], []
    for style in STYLES:
        payload = torch.load(pgt_root() / "dino_224" / f"{style}.pt", map_location="cpu", weights_only=False)
        all_cls.append(payload["cls"].float())
        all_ids.extend(payload["ids"])
    cls_all = F.normalize(torch.cat(all_cls, 0), dim=-1)
    pos_in_cls = {id_to_idx[i]: k for k, i in enumerate(all_ids)}
    train_items = [id_to_idx[i] for i in all_ids if items[id_to_idx[i]]["split"] == "train"]
    out = {}
    for style in STYLES:
        src_idx = [i for i, it in enumerate(items) if it["style"] == style and it["split"] == "train"]
        if not src_idx:
            continue
        src_vecs = cls_all[[pos_in_cls[i] for i in src_idx]]
        by_target = {}
        for tgt in STYLES:
            if tgt == style:
                continue
            donors_t = sorted(i for i in train_items if items[i]["style"] == tgt)
            if not donors_t:
                continue
            donor_vecs = cls_all[[pos_in_cls[i] for i in donors_t]]
            sims = src_vecs @ donor_vecs.t()
            topv, topi = sims.topk(min(topk, len(donors_t)), dim=1)
            by_target[tgt] = {
                "donor_ids": [[items[donors_t[j]]["id"] for j in row] for row in topi.tolist()],
                "donor_scores": topv.tolist(),
            }
            del donor_vecs, sims
            print(f"[retrieval] {style}->{tgt}: {len(src_idx)} 源 × top{topk}", flush=True)
        out[style] = {"source_ids": [items[i]["id"] for i in src_idx], "by_target": by_target}
    payload = {"topk": topk, "styles": out}
    (out_dir / "top8.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"styles": list(out), "topk": topk}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "manifest", "latents", "dino", "retrieval"], default="all")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    root = work_root()
    root.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    if args.stage in {"all", "manifest"}:
        items = enumerate_images()
        print(f"[manifest] {len(items)} 张，train={sum(1 for it in items if it['split']=='train')} val={sum(1 for it in items if it['split']=='val')}")
        save_manifest(items)
    items = load_manifest()
    if args.stage in {"all", "latents"}:
        build_mean_latents(items, batch=args.batch, resume=resume)
    if args.stage in {"all", "dino"}:
        build_dino(items, resume=resume)
    if args.stage in {"all", "retrieval"}:
        build_retrieval(items)


if __name__ == "__main__":
    main()
