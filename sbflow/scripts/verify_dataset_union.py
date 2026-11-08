#!/usr/bin/env python3
"""Verify the D5 latent cache is the disjoint union of wikiart_new_512 and distinct5_512/train."""
import glob
import os
from pathlib import Path

import torch

REPO = Path("/mnt/i/Latent_Style/WEAVE")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

def files(d: Path) -> set:
    if not d.exists():
        return set()
    return {p.name for p in d.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".webp")}

caches = sorted(glob.glob("/mnt/i/Latent_Style/**/pseudo_gt_24h/orig_mean_fp16/*.pt", recursive=True))
print("cache files:", [os.path.basename(c) for c in caches])
print()

total_cache = 0
print(f"{'style':20s} {'wikiart_new':>12s} {'distinct5':>10s} {'sum':>7s} {'cache_n':>8s} {'match':>6s} {'overlap':>8s}")
for i, s in enumerate(STYLES):
    a = files(REPO / "datasets/wikiart_new_512" / s)
    b = files(REPO / "datasets/distinct5_512/train" / s)
    cache = next((c for c in caches if Path(c).stem.endswith(s)), None)
    n = None
    if cache:
        d = torch.load(cache, map_location="cpu", weights_only=False)
        for k in ("latent", "z", "latents", "data"):
            if isinstance(d, dict) and k in d:
                d = d[k]
                break
        n = int(d.shape[0]) if hasattr(d, "shape") else None
    total_cache += n or 0
    print(f"{s:20s} {len(a):12d} {len(b):10d} {len(a)+len(b):7d} {str(n):>8s} "
          f"{'YES' if n == len(a)+len(b) else 'NO':>6s} {len(a & b):8d}")

print(f"\nTOTAL cache rows = {total_cache}")
print(f"TOTAL images (wikiart_new + distinct5/train) = "
      f"{sum(len(files(REPO / 'datasets/wikiart_new_512' / s)) + len(files(REPO / 'datasets/distinct5_512/train' / s)) for s in STYLES)}")

# Do the caches carry ids we can audit against?
for c in caches:
    d = torch.load(c, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        keys = [k for k in d.keys() if k not in ("latent", "z", "latents", "data")]
        print(f"\n{os.path.basename(c)} keys={list(d.keys())[:8]}")
        if "ids" in d:
            print(f"  n_ids={len(d['ids'])} sample={d['ids'][0]}")
