"""WEAVE Art-VAE Suite: High-Speed Pre-resizer & Memory-Mapped RAW Binary Packer.

Packs the full 103,250 WikiArt images into a single continuous memory-mapped binary:
- File: /root/Workspace/data/wikiart_512_raw.bin (uint8 array of shape [N, 512, 512, 3], ~75 GB).
- Resides seamlessly in 251GB RAM via Linux OS Page Cache / memmap.
- Eliminates 100% of runtime CPU JPEG decode and Bicubic resize bottlenecks.
- Robust against truncated images (uses ImageFile.LOAD_TRUNCATED_IMAGES = True).
- Outputs metadata manifest: wikiart_512_raw_manifest.json with frozen 2048 val split.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[2]


def process_single_image(args: Tuple[int, str, str, str, str, str]) -> Tuple[int, Optional[bytes], dict]:
    """Worker function to read, resize, and convert a single image to raw 512x512x3 uint8 bytes."""
    idx, img_path_str, filename, style, artist, genre = args
    try:
        p = Path(img_path_str)
        if not p.exists():
            return idx, None, {}
        im = Image.open(p).convert("RGB")
        im_res = im.resize((512, 512), Image.BICUBIC)
        raw_bytes = np.array(im_res, dtype=np.uint8).tobytes()
        meta = {
            "filename": filename,
            "style": style,
            "artist": artist,
            "genre": genre,
        }
        return idx, raw_bytes, meta
    except Exception as e:
        return idx, None, {}


def pack_dataset(
    csv_path: Path,
    image_roots: List[Path],
    dest_dir: Path,
    num_workers: int = 16,
    chunk_size: int = 2000,
):
    dest_dir.mkdir(parents=True, exist_ok=True)
    bin_path = dest_dir / "wikiart_512_raw.bin"
    manifest_path = dest_dir / "wikiart_512_raw_manifest.json"

    print("=== [Packing WikiArt into 512x512 Memory-Mapped RAW Binary] ===", flush=True)

    # 1. Index available images
    print("[1/4] Indexing image paths on disk...", flush=True)
    img_lookup = {}
    for r in image_roots:
        if r.exists():
            for p in r.rglob("*.jpg"):
                img_lookup[p.name] = str(p)
    print(f"  Indexed {len(img_lookup)} image paths.", flush=True)

    # 2. Parse CSV
    print("[2/4] Parsing metadata CSV...", flush=True)
    styles_set, artists_set, genres_set = set(), set(), set()
    raw_candidates = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("new_filename", "").strip()
            st = row.get("style", "").strip()
            ar = row.get("artist", "").strip()
            ge = row.get("genre", "").strip()
            in_tr = row.get("in_train", "").strip().lower() == "true"

            if not fname or not st or fname not in img_lookup:
                continue

            styles_set.add(st)
            if ar:
                artists_set.add(ar)
            if ge:
                genres_set.add(ge)

            raw_candidates.append((
                len(raw_candidates),
                img_lookup[fname],
                fname,
                st,
                ar,
                ge,
                in_tr,
            ))

    style_to_id = {s: i for i, s in enumerate(sorted(styles_set))}
    artist_to_id = {a: i for i, a in enumerate(sorted(artists_set))}
    genre_to_id = {g: i for i, g in enumerate(sorted(genres_set))}

    print(f"  Total valid candidates with images: {len(raw_candidates)}", flush=True)
    print(f"  Classes: {len(style_to_id)} styles, {len(artist_to_id)} artists, {len(genre_to_id)} genres.", flush=True)

    # 3. Parallel Resize and Pack into Binary
    print(f"[3/4] Packing images into {bin_path} with {num_workers} processes...", flush=True)
    bytes_per_image = 512 * 512 * 3 # 786,432 bytes

    valid_manifest = []
    t0 = time.time()

    with open(bin_path, "wb") as f_bin:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for c_start in range(0, len(raw_candidates), chunk_size):
                chunk = raw_candidates[c_start : c_start + chunk_size]
                worker_inputs = [
                    (idx, p_str, fname, st, ar, ge)
                    for (idx, p_str, fname, st, ar, ge, _) in chunk
                ]
                results = list(executor.map(process_single_image, worker_inputs))

                for j, (orig_idx, raw_bytes, meta) in enumerate(results):
                    if raw_bytes is not None and len(raw_bytes) == bytes_per_image:
                        in_tr = chunk[j][6]
                        f_bin.write(raw_bytes)
                        cur_idx = len(valid_manifest)
                        valid_manifest.append({
                            "raw_index": cur_idx,
                            "filename": meta["filename"],
                            "style": meta["style"],
                            "style_id": style_to_id[meta["style"]],
                            "artist": meta["artist"],
                            "artist_id": artist_to_id.get(meta["artist"], -1),
                            "genre": meta["genre"],
                            "genre_id": genre_to_id.get(meta["genre"], -1),
                            "in_train": in_tr,
                        })

                elapsed = time.time() - t0
                speed = len(valid_manifest) / max(elapsed, 1e-4)
                pct = (c_start + len(chunk)) / len(raw_candidates) * 100.0
                print(
                    f"\r  Packed {len(valid_manifest):6d}/{len(raw_candidates)} images ({pct:5.1f}%) | "
                    f"Speed: {speed:5.1f} img/s | Size: {len(valid_manifest)*bytes_per_image/(1024**3):.2f} GB",
                    end="", flush=True
                )

    total_packed = len(valid_manifest)
    total_sec = time.time() - t0
    final_gb = (total_packed * bytes_per_image) / (1024**3)
    print(f"\n[Done] Successfully packed {total_packed} images ({final_gb:.2f} GB) in {total_sec:.1f}s ({total_packed/total_sec:.1f} img/s)!", flush=True)

    # 4. Partition Frozen 2048 Val Set & Save Manifest
    print("[4/4] Freezing 2048 validation set and writing manifest...", flush=True)
    # Select 2048 images where in_train == False
    test_split = [r for r in valid_manifest if not r["in_train"]]
    import random
    rng = random.Random(42)
    rng.shuffle(test_split)
    val_set = test_split[:2048]
    val_indices = {r["raw_index"] for r in val_set}

    train_set = [r for r in valid_manifest if r["raw_index"] not in val_indices]

    manifest_payload = {
        "bin_path": str(bin_path),
        "total_images": total_packed,
        "image_size": 512,
        "bytes_per_image": bytes_per_image,
        "shape": [total_packed, 512, 512, 3],
        "dtype": "uint8",
        "num_styles": len(style_to_id),
        "style_to_id": style_to_id,
        "artist_to_id": artist_to_id,
        "genre_to_id": genre_to_id,
        "num_val_images": len(val_set),
        "num_train_images": len(train_set),
        "val_indices": sorted(list(val_indices)),
        "records": valid_manifest,
    }

    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Done] Wrote manifest to {manifest_path} (Train: {len(train_set)}, Val: {len(val_set)})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/root/Workspace/data/wikiart/all_data_info.csv")
    ap.add_argument("--roots", default="/root/Workspace/data/wikiart/train,/root/Workspace/data/wikiart/test")
    ap.add_argument("--dest", default="/root/Workspace/data/wikiart_512_raw")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pack_dataset(
        Path(args.csv),
        [Path(p) for p in args.roots.split(",")],
        Path(args.dest),
        num_workers=args.workers,
    )
