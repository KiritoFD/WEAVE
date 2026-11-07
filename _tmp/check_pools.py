import os
from pathlib import Path

base = Path("/mnt/i/Latent_Style/WEAVE/results/D5-512")
datasets_root = Path("/mnt/i/Latent_Style/WEAVE/datasets")

# Collect candidate source pools
pools = {
    "data_test": Path("/mnt/i/Latent_Style/WEAVE/data/test"),
    "distinct5_test": datasets_root / "distinct5_512" / "test",
    "distinct5_train": datasets_root / "distinct5_512" / "train",
    "zstar_src": datasets_root / "distinct5_512_train_zstar_src",
    "wikiart512_test": datasets_root / "wikiart512_5style" / "test",
    "wikiart512_train": datasets_root / "wikiart512_5style" / "train",
}

pool_indices = {}
for pname, pdir in pools.items():
    if pdir.exists():
        idx = {}
        for p in pdir.rglob("*.jpg"):
            idx[p.stem] = p
            # also short stem
            short = p.stem.split("__")[-1]
            idx[short] = p
        pool_indices[pname] = idx
        print(f"Pool {pname:20s}: {len(idx)} entries")

methods = ["stylealigned", "zstar", "seedream"]

for m in methods:
    mdir = base / m
    imgs = list(mdir.glob("*.png"))
    print(f"\nChecking method: {m} ({len(imgs)} imgs)")
    for pname, idx in pool_indices.items():
        matched = 0
        for im in imgs:
            # extract stem between style and target
            stem = im.stem
            for cand in idx:
                if cand in stem:
                    matched += 1
                    break
        print(f"  match against {pname:20s}: {matched}/{len(imgs)}")
