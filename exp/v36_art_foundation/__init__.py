"""V36 CLEAVE-One: Full-WikiArt 131-Movement Art Foundation Model (Zero-DINO + 45M P-Core + S10).

Strict invariants across the entire package:
1. Random initialization for P-Core (~45.2M, V34 C2 208-ch topology) and Style MLP (1561->512->256, <1M);
   ZERO weights inherited from V32/V34/V35 closed-set checkpoints.
2. Whole-WikiArt deduplication before movement isolation; 5 held-out movements (default Core-5)
   are strictly excluded from Seen-Train, Seen-Dev, Artist-OOD, PCA Q32 fitting, luma projection,
   feature normalization stats, and checkpoint selection.
3. Zero DINOv2 / CSD / CLIP / LPIPS in data prep, Style MLP, P-Core, or training losses.
4. Decoding always routes through official S10 contract (src.utils.load_s10_decoder / decode_s10).
"""
from __future__ import annotations

import os
from pathlib import Path

import paths
from src.utils.runtime import REPO as ROOT

OUT = ROOT / "runs/v36_art_foundation"
CACHE_DIR = OUT / "cache"
CKPT_DIR = OUT / "checkpoints"
MEAS_DIR = OUT / "measured"
FIG_DIR = OUT / "figures"
LOG_DIR = OUT / "logs"

SEED_V36 = 20263600

CORE_5_SLUGS = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]

HOLDOUT_PRESETS: dict[str, list[str]] = {
    "core5": [
        "Early Renaissance",
        "Impressionism",
        "Minimalism",
        "Rococo",
        "Ukiyo-e",
    ],
    "v35": [
        "Pointillism",
        "Cubism",
        "Pop Art",
        "Art Nouveau (Modern)",
        "Fauvism",
    ],
}


def find_wikiart_data_root() -> Path:
    """Resolve WikiArt all_data_info.csv + train/test root without hardcoded drive paths."""
    env_p = os.environ.get("WEAVE_WIKIART_ROOT", "").strip()
    cands = []
    if env_p:
        cands.append(Path(env_p))
    cands.extend([
        paths.datasets_root("wikiart_raw_full"),
        paths.asset_root() / "Latent_Style/Data",
        paths.asset_root() / "Data",
        ROOT.parent / "Latent_Style/Data",
    ])
    for p in cands:
        if (p / "all_data_info.csv").exists():
            return p
    raise FileNotFoundError(f"WikiArt Data root (all_data_info.csv) not found in {cands}")


__all__ = [
    "CACHE_DIR",
    "CKPT_DIR",
    "CORE_5_SLUGS",
    "FIG_DIR",
    "HOLDOUT_PRESETS",
    "LOG_DIR",
    "MEAS_DIR",
    "OUT",
    "ROOT",
    "SEED_V36",
    "find_wikiart_data_root",
]
