"""Real-painting dataset for V11-B wavelet flow training.

Painting-level train/holdout split by stem hash (no patch leakage).
Returns 128x128 crops with their 2-level Haar bands, Sobel edge map of
LL2, and style id. All band math in fp32 on GPU-friendly tensors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bands import dwt2, sobel_mag

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo",
          "Ukiyo_e"]
STYLE2ID = {s: i for i, s in enumerate(STYLES)}

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / \
    "distinct5_512" / "train"


def _is_holdout(stem: str, frac: float = 0.1) -> bool:
    h = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16)
    return (h % 10000) < frac * 10000


class RealWaveletDataset(Dataset):
    """128x128 random crops from 512px real paintings.

    split: 'train' | 'holdout' (painting-level, hash on file stem).
    Returns dict with crop uint8->float /255, bands, edge, style_id.
    """

    def __init__(self, split: str = "train", crop: int = 128,
                 holdout_frac: float = 0.1, seed: int = 0):
        assert split in ("train", "holdout")
        self.crop = crop
        self.items = []  # (path, style_id, stem)
        for style in STYLES:
            d = DATA_ROOT / style
            for p in sorted(d.glob("*.jpg")):
                stem = p.stem
                if _is_holdout(stem, holdout_frac) == (split == "holdout"):
                    self.items.append((p, STYLE2ID[style], stem))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, sid, _ = self.items[idx]
        im = Image.open(path).convert("RGB")
        w, h = im.size
        c = self.crop
        if w < c or h < c:
            im = im.resize((max(w, c), max(h, c)), Image.BILINEAR)
            w, h = im.size
        x0 = int(self.rng.integers(0, w - c + 1))
        y0 = int(self.rng.integers(0, h - c + 1))
        crop = torch.from_numpy(
            np.asarray(im.crop((x0, y0, x0 + c, y0 + c)),
                       dtype=np.float32) / 255.0).permute(2, 0, 1)
        return crop, sid


class RamWaveletDataset(Dataset):
    """Same split/crop logic as RealWaveletDataset but preloads all
    training images into RAM as uint8 [N,3,512,512] once (~3.5GB),
    removing PIL decode from the training loop entirely."""

    def __init__(self, split: str = "train", crop: int = 128,
                 holdout_frac: float = 0.1, seed: int = 0):
        base = RealWaveletDataset(split, crop, holdout_frac, seed)
        self.crop = crop
        self.items = base.items
        self.sids = torch.tensor([s for _, s, _ in self.items],
                                 dtype=torch.long)
        n = len(self.items)
        self.buf = torch.empty(n, 3, 512, 512, dtype=torch.uint8)
        for i, (p, _, _) in enumerate(self.items):
            im = Image.open(p).convert("RGB")
            self.buf[i] = torch.from_numpy(
                np.asarray(im, dtype=np.uint8).transpose(2, 0, 1))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        c = self.crop
        x0 = int(self.rng.integers(0, 512 - c + 1))
        y0 = int(self.rng.integers(0, 512 - c + 1))
        crop = self.buf[idx, :, y0:y0 + c, x0:x0 + c].float() / 255.0
        return crop, int(self.sids[idx])


def collate_batch(batch):
    """CPU collate -> move to device, compute bands once.

    Returns dict: LL2,H2,H1 (targets), LL2c (cond), edge, style_id.
    Bands are computed on the full crop in fp32.
    """
    crops = torch.stack([b[0] for b in batch])
    sids = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return crops, sids


def make_batch(crops: torch.Tensor, sids: torch.Tensor, device="cuda"):
    """crops [B,3,128,128] float 0-1 -> band dict on device (fp32)."""
    crops = crops.to(device)
    d = dwt2(crops)
    return {
        "H1": d["H1"].to(device), "H2": d["H2"].to(device),
        "LL2": d["LL2"].to(device), "LL1": d["LL1"].to(device),
        "edge1": sobel_mag(d["LL1"].mean(1, keepdim=True)).to(device),
        "edge2": sobel_mag(d["LL2"].mean(1, keepdim=True)).to(device),
        "style_id": sids.to(device),
    }
