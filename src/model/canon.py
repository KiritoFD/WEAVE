"""Content canonicalizer: the frozen c1_gentle content view.

Both the start model and every edit objective consume this same view, so the
canonicalizer is loaded from one function here to guarantee a single instance
and a single set of statistics.
"""
from __future__ import annotations

from pathlib import Path

import torch

from tools.v6.canonicalize import ContentCanonicalizer  # noqa: F401

REPO = Path(__file__).resolve().parents[2]
CANON_STATS = REPO / "runs/v6/cache/canonicalizer_stats.pt"


def canonicalizer(device: str = "cuda", *, mode: str = "c1_gentle"):
    """Load the frozen canonicalizer once. Statistics file must already exist."""
    if not CANON_STATS.exists():
        raise FileNotFoundError(f"canonicalizer statistics missing: {CANON_STATS}")
    stats = torch.load(CANON_STATS, map_location="cpu", weights_only=False)
    model = ContentCanonicalizer(stats).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


__all__ = ["ContentCanonicalizer", "canonicalizer", "CANON_STATS"]
