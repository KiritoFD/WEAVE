import json
import os
from pathlib import Path

ROOTS = {
    "D5-512": Path("/mnt/i/Latent_Style/WEAVE/results/D5-512"),
    "P256": Path("/mnt/i/Latent_Style/WEAVE/results/P256"),
    "R5": Path("/mnt/i/Latent_Style/WEAVE/results/R5"),
    "R5-512": Path("/mnt/i/Latent_Style/WEAVE/results/R5-512"),
    "R5-WikiArt": Path("/mnt/i/Latent_Style/WEAVE/results/R5-WikiArt"),
}

out = {}
for name, root in ROOTS.items():
    if not root.exists():
        out[name] = "MISSING"
        continue
    subs = sorted(p for p in root.iterdir() if p.is_dir())
    entry = {}
    for d in subs:
        imgs = [p for p in d.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        total = sum(p.stat().st_size for p in imgs)
        entry[d.name] = {"n": len(imgs), "MB": round(total / 1048576, 1)}
    out[name] = entry

print(json.dumps(out, indent=2))
print("\n=== metrics csv locations ===")
for pat in ["eval_outputs/*/*/metrics.csv", "eval_outputs/*/*/*.csv", "results/*/*.csv"]:
    for p in sorted(Path("/mnt/i/Latent_Style/WEAVE").glob(pat)):
        print(f"  {p.relative_to('/mnt/i/Latent_Style/WEAVE')}  {p.stat().st_size/1024:.0f} KB")
