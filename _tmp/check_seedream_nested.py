"""Count nested images for methods whose files live in style subdirectories."""
from pathlib import Path

REPO = Path("/mnt/i/Latent_Style/WEAVE")
CASES = [
    ("P2A-256", REPO / "results/P256/seedream"),
    ("R5-WikiArt", REPO / "results/R5-WikiArt/seedream"),
    ("D5-512", REPO / "results/D5-512/seedream"),
]
for ds, root in CASES:
    if not root.is_dir():
        print(f"{ds}/seedream MISSING")
        continue
    flat = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    nested = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.parent != root]
    print(f"{ds}/seedream: flat={len(flat)} nested={len(nested)}")
    subs = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"   subdirs: {subs}")
    for s in subs[:6]:
        d = root / s
        n = len([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        sample = next((p.name for p in d.rglob("*") if p.suffix.lower() == ".png"), None)
        print(f"     {s}: {n} imgs  sample={sample}")
