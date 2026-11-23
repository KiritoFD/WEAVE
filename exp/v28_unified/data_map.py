"""100% deterministic RGB path resolution for train_4400 (fixing the 87/150 missing mapping)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exp.v18 import STYLES, canonical_pid
from exp.v28_unified import ROOT


def _scan_style_dirs(root: Path, index: dict[tuple[int, str], Path]) -> None:
    if not root.exists():
        return
    for sid, cls in enumerate(STYLES):
        sub = root / cls
        if not sub.exists():
            continue
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                index.setdefault((sid, canonical_pid(p.name)), p)


def build_rgb_lookup() -> dict[tuple[int, str], Path]:
    """Build complete (sid, canonical_pid) -> Path lookup over distinct5 + wikiart assets."""
    d5_map: dict[tuple[int, str], Path] = {}
    for cand in (
        ROOT.parent / "train_512clean",
        ROOT / "datasets/distinct5_512/train",
        ROOT / "datasets/distinct5_512/train_512clean",
    ):
        _scan_style_dirs(cand, d5_map)

    wiki_map: dict[tuple[int, str], Path] = {}
    _scan_style_dirs(ROOT / "datasets/wikiart_new_512", wiki_map)

    manifest_p = ROOT / "weights_local/wikiart_new_512/manifest.jsonl"
    raw_wiki_root = ROOT.parent / "data/wikiart"
    if manifest_p.exists() and raw_wiki_root.exists():
        with manifest_p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = STYLES.index(row["style"])
                sub = "train/train" if row.get("provenance") == "official_train" else "test/test"
                rid = row["id"]
                p = raw_wiki_root / sub / f"{rid}.jpg"
                if p.exists():
                    wiki_map.setdefault((sid, canonical_pid(row["filename"])), p)

    merged = dict(wiki_map)
    merged.update(d5_map)
    return merged


def resolve_train_rgb_paths(
    ids: list[str],
    sids: list[int] | torch.Tensor,
    *,
    require_full: bool = True,
) -> list[Path | None]:
    """Map every train sample id to its original RGB image path; raise if require_full and any miss."""
    lookup = build_rgb_lookup()
    d5_only: dict[tuple[int, str], Path] = {}
    for cand in (
        ROOT.parent / "train_512clean",
        ROOT / "datasets/distinct5_512/train",
        ROOT / "datasets/distinct5_512/train_512clean",
    ):
        _scan_style_dirs(cand, d5_only)

    sid_list = sids.tolist() if torch.is_tensor(sids) else list(sids)
    resolved: list[Path | None] = []
    missing: list[tuple[int, str]] = []
    for sid, raw_id in zip(sid_list, ids):
        key = (int(sid), canonical_pid(raw_id))
        if str(raw_id).startswith("d5__") and key in d5_only:
            resolved.append(d5_only[key])
        elif key in lookup:
            resolved.append(lookup[key])
        else:
            resolved.append(None)
            missing.append((int(sid), str(raw_id)))
    if require_full and missing:
        raise RuntimeError(
            f"Incomplete train RGB mapping: {len(missing)}/{len(ids)} missing (first 5: {missing[:5]})"
        )
    return resolved


def load_rgb_512_tensor(paths: list[Path]) -> torch.Tensor:
    """Load RGB images as float32 (N, 3, 512, 512) in [0, 1] matching latent build_assets resize."""
    arrs = []
    for p in paths:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            if rgb.size != (512, 512):
                rgb = rgb.resize((512, 512), Image.BILINEAR)
            arrs.append(np.asarray(rgb, dtype=np.uint8).copy())
    t = torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    return t
