import os
from pathlib import Path

base = Path("/mnt/i/Latent_Style/WEAVE/results/D5-512")
if not base.exists():
    print(f"Base {base} does not exist")
    exit(1)

subdirs = sorted([d for d in base.iterdir() if d.is_dir()])
print(f"Found {len(subdirs)} subdirectories in {base}:")
for d in subdirs:
    pngs = list(d.glob("*.png"))
    jpgs = list(d.glob("*.jpg"))
    csvs = list(d.glob("*.csv"))
    jsons = list(d.glob("*.json"))
    sample_name = pngs[0].name if pngs else (jpgs[0].name if jpgs else "none")
    print(f"  {d.name:20s}: {len(pngs)} pngs, {len(jpgs)} jpgs, {len(csvs)} csvs, {len(jsons)} jsons | sample: {sample_name}")
