"""Scan all available WikiArt image datasets and inventory on Drive I."""
import os
from pathlib import Path
import csv

roots_to_check = [
    Path("/mnt/i/Latent_Style/Data"),
    Path("/mnt/i/Latent_Style/WEAVE/datasets"),
    Path("/mnt/i/datasets"),
    Path("/mnt/i/Latent_Style/WEAVE/data"),
]

print("=== WikiArt Image Dataset Inventory Scan ===")
exts = {".jpg", ".jpeg", ".png", ".webp"}

for r in roots_to_check:
    if not r.exists():
        print(f"Skipping non-existent root: {r}")
        continue
    print(f"\nScanning: {r}")
    subdirs = [d for d in r.iterdir() if d.is_dir()]
    total_r = 0
    for d in sorted(subdirs):
        # count images recursively or shallowly
        imgs = [p for p in d.rglob("*") if p.suffix.lower() in exts]
        print(f"  [{d.name:30s}]: {len(imgs):6d} images")
        total_r += len(imgs)
    print(f"  -> Total under {r.name}: {total_r} images")

# Check all_data_info.csv
csv_paths = [
    Path("/mnt/i/Latent_Style/Data/all_data_info.csv"),
    Path("/mnt/i/Latent_Style/all_data_info.csv"),
]
for cp in csv_paths:
    if cp.is_file():
        with open(cp, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = sum(1 for _ in reader)
        print(f"\nMetadata CSV: {cp} has {rows} records (Header: {header[:5]}...)")

# Check Images.rar
rar_path = Path("/mnt/i/Images.rar")
if rar_path.exists():
    sz_gb = rar_path.stat().st_size / (1024**3)
    print(f"\nImages.rar archive found: {sz_gb:.2f} GB at {rar_path}")
