from pathlib import Path

train_dir = Path("/mnt/i/Latent_Style/WEAVE/datasets/distinct5_512/train")
train_index = {}
for p in train_dir.rglob("*.jpg"):
    train_index[p.stem] = p
    short = p.stem.split("__")[-1]
    train_index[short] = p

print(f"train_index size: {len(train_index)}")

for m in ["stylealigned", "zstar"]:
    mdir = Path(f"/mnt/i/Latent_Style/WEAVE/results/D5-512/{m}")
    imgs = list(mdir.glob("*.png"))
    matched = 0
    for im in imgs:
        stem = im.stem
        # parse target
        tgt = stem.split("__to__")[-1]
        left = stem.split("__to__")[0]
        # strip all style prefixes
        for s in ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]:
            while left.startswith(s + "__") or left.startswith(s + "_"):
                left = left[len(s)+2:] if left.startswith(s + "__") else left[len(s)+1:]
        
        # now left is the pure stem
        if left in train_index or f"{im.stem.split('__')[0]}__{left}" in train_index:
            matched += 1
    print(f"{m}: {matched}/{len(imgs)} matched against distinct5_512/train")
