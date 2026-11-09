"""Exact inventory: the 13 paper baselines x 3 datasets, flat image counts only."""
from pathlib import Path

REPO = Path("/mnt/i/Latent_Style/WEAVE")
METHODS = ["identity", "adain", "wct", "cut", "samst", "samam", "sdturbo",
           "styleid", "styleshot", "stytr2", "aespa", "weave", "seedream"]
DATASETS = {
    "D5-512": REPO / "results" / "D5-512",
    "P2A-256": REPO / "results" / "P256",
    "R5-WikiArt": REPO / "results" / "R5-WikiArt",
}

print(f"{'method':12s}" + "".join(f"{d:>22s}" for d in DATASETS))
print("-" * (12 + 22 * len(DATASETS)))
tot = {}
for m in METHODS:
    line = f"{m:12s}"
    for d, root in DATASETS.items():
        md = root / m
        if not md.is_dir():
            line += f"{'MISSING':>22s}"
            continue
        flat = [p for p in md.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        size = sum(p.stat().st_size for p in flat) / 1048576
        tot[d] = tot.get(d, 0) + size
        line += f"{len(flat):6d} imgs {size:8.0f}MB".rjust(22)
    print(line)
print("-" * (12 + 22 * len(DATASETS)))
print("totals MB:", {k: round(v) for k, v in tot.items()})
print("\nsubdirs present per method (nested duplicates are excluded above):")
for m in METHODS:
    for d, root in DATASETS.items():
        md = root / m
        if md.is_dir():
            subs = [p.name for p in md.iterdir() if p.is_dir()]
            if subs:
                print(f"  {d}/{m}: nested {subs}")
