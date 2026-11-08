#!/usr/bin/env python3
"""Check for duplicate ids inside each D5 latent cache and across the two source pools."""
import glob
import os
from collections import Counter
from pathlib import Path

import torch

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
caches = sorted(glob.glob("/mnt/i/Latent_Style/**/pseudo_gt_24h/orig_mean_fp16/*.pt", recursive=True))

grand_dup = 0
for c in caches:
    d = torch.load(c, map_location="cpu", weights_only=False)
    ids = list(d["ids"])
    cnt = Counter(ids)
    dup = {k: v for k, v in cnt.items() if v > 1}
    extra = sum(v - 1 for v in dup.values())
    grand_dup += extra
    print(f"{os.path.basename(c):32s} n={len(ids):6d} unique={len(cnt):6d} duplicated_ids={len(dup):4d} extra_rows={extra:4d}")
    for k, v in list(dup.items())[:3]:
        print(f"    x{v}  {k}")
print(f"\nTOTAL duplicate extra rows = {grand_dup}")

# Cross-pool filename overlap already measured; confirm the ids collide for those files.
REPO = Path("/mnt/i/Latent_Style/WEAVE")
for s in STYLES:
    a = {p.name for p in (REPO / "datasets/wikiart_new_512" / s).iterdir() if p.suffix.lower() in (".jpg", ".png")}
    b = {p.name for p in (REPO / "datasets/distinct5_512/train" / s).iterdir() if p.suffix.lower() in (".jpg", ".png")}
    print(f"{s:20s} filename_overlap={len(a & b)}")
