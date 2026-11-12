"""WEAVE Art-VAE Suite: Dataset & Metadata Utilities.

Parses all_data_info.csv to extract ~137 style classes, builds frozen 2048-image
validation set (disjoint from training), and provides PyTorch Dataset/DataLoader
with metadata (style_id, artist_id, genre_id).
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = REPO_ROOT / "eval_protocol"


def load_wikiart_metadata(
    csv_path: Path,
    image_roots: List[Path],
) -> Tuple[List[dict], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Parse all_data_info.csv and match with available image files."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    # Build image lookup index by filename across provided image roots
    img_lookup = {}
    for r in image_roots:
        if r.exists():
            for p in r.rglob("*.jpg"):
                img_lookup[p.name] = p

    styles_set = set()
    artists_set = set()
    genres_set = set()

    raw_records = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("new_filename", "").strip()
            st = row.get("style", "").strip()
            ar = row.get("artist", "").strip()
            ge = row.get("genre", "").strip()
            in_tr = row.get("in_train", "").strip().lower() == "true"

            if not fname or not st:
                continue

            styles_set.add(st)
            if ar:
                artists_set.add(ar)
            if ge:
                genres_set.add(ge)

            raw_records.append({
                "filename": fname,
                "style": st,
                "artist": ar,
                "genre": ge,
                "in_train": in_tr,
                "path": img_lookup.get(fname),
            })

    # Sort classes canonically to ensure fixed integer IDs
    style_to_id = {s: i for i, s in enumerate(sorted(styles_set))}
    artist_to_id = {a: i for i, a in enumerate(sorted(artists_set))}
    genre_to_id = {g: i for i, g in enumerate(sorted(genres_set))}

    for r in raw_records:
        r["style_id"] = style_to_id[r["style"]]
        r["artist_id"] = artist_to_id.get(r["artist"], -1)
        r["genre_id"] = genre_to_id.get(r["genre"], -1)

    return raw_records, style_to_id, artist_to_id, genre_to_id


def build_frozen_val2048(
    records: List[dict],
    output_path: Path,
    n_val: int = 2048,
    seed: int = 42,
) -> List[dict]:
    """Select exactly n_val independent images from test split, frozen by seed."""
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))

    # Prefer records marked as in_train == False and with resolvable image file
    candidates = [r for r in records if not r["in_train"] and r["path"] is not None]
    if len(candidates) < n_val:
        # Fallback to any valid records with fixed seed
        candidates = [r for r in records if r["path"] is not None]

    rng = random.Random(seed)
    rng.shuffle(candidates)
    val_set = candidates[:n_val]

    # Save minimal frozen manifest
    serializable = [
        {
            "filename": r["filename"],
            "style": r["style"],
            "style_id": r["style_id"],
            "artist": r["artist"],
            "artist_id": r["artist_id"],
            "genre": r["genre"],
            "genre_id": r["genre_id"],
            "rel_path": str(r["path"]) if r["path"] else r["filename"],
        }
        for r in val_set
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Dataset] Wrote frozen validation manifest ({len(serializable)} items) to {output_path}")
    return serializable


class WikiArtDataset(Dataset):
    """PyTorch Dataset for WikiArt images scaled to 512x512 in [-1, 1]."""

    def __init__(
        self,
        records: List[dict],
        image_size: int = 512,
        is_training: bool = True,
    ):
        self.records = [r for r in records if r.get("path") and Path(r["path"]).exists()]
        self.image_size = image_size
        self.is_training = is_training

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        p = Path(r["path"])
        img = Image.open(p).convert("RGB")
        tensor = self.transform(img)

        return {
            "image": tensor,
            "style_id": torch.tensor(r["style_id"], dtype=torch.long),
            "artist_id": torch.tensor(r.get("artist_id", -1), dtype=torch.long),
            "genre_id": torch.tensor(r.get("genre_id", -1), dtype=torch.long),
            "filename": r["filename"],
        }


class RawMemmapWikiArtDataset(Dataset):
    """Ultra-fast zero-decode, zero-resize memory-mapped dataset.

    Resides in 251GB RAM. Fetches pre-resized 512x512x3 uint8 array directly
    from shared memory / memmap with sub-millisecond latency.
    """

    def __init__(self, manifest_path: Path, is_training: bool = True):
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.bin_path = Path(self.manifest["bin_path"])
        self.shape = tuple(self.manifest["shape"])
        self.mmap = np.memmap(str(self.bin_path), dtype=np.uint8, mode="r", shape=self.shape)

        val_set = set(self.manifest["val_indices"])
        if is_training:
            self.indices = [r["raw_index"] for r in self.manifest["records"] if r["raw_index"] not in val_set]
        else:
            self.indices = [r["raw_index"] for r in self.manifest["records"] if r["raw_index"] in val_set]

        self.records_by_idx = {r["raw_index"]: r for r in self.manifest["records"]}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        raw_idx = self.indices[idx]
        # Direct zero-copy slice from RAM (512, 512, 3)
        arr = self.mmap[raw_idx]
        # Fast normalize to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        rec = self.records_by_idx[raw_idx]

        return {
            "image": tensor,
            "style_id": torch.tensor(rec["style_id"], dtype=torch.long),
            "artist_id": torch.tensor(rec.get("artist_id", -1), dtype=torch.long),
            "genre_id": torch.tensor(rec.get("genre_id", -1), dtype=torch.long),
            "filename": rec["filename"],
        }


class ValDataset2048(Dataset):
    """Validation dataset with automatic zero-decode memory-map support."""

    def __init__(self, manifest_path: Path, image_size: int = 512):
        self.manifest_path = Path(manifest_path)
        if "raw_manifest" in self.manifest_path.name or "wikiart_512_raw" in str(self.manifest_path):
            self.dataset = RawMemmapWikiArtDataset(self.manifest_path, is_training=False)
            self.use_raw = True
            self.items = [self.dataset.records_by_idx[idx] for idx in self.dataset.indices]
        else:
            self.use_raw = False
            self.items = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.image_size = image_size
            self.tf = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        if self.use_raw:
            return self.dataset[idx]
        item = self.items[idx]
        p = Path(item["rel_path"])
        if not p.is_absolute():
            p = REPO_ROOT / p
        img = Image.open(p).convert("RGB")
        tensor = self.tf(img)
        return {
            "image": tensor,
            "filename": item["filename"],
            "style": item["style"],
            "style_id": item["style_id"],
        }
