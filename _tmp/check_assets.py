import sys
from collections import Counter

import torch

sys.path.insert(0, "/mnt/i/Latent_Style/WEAVE")
from tools.pseudo_gt.build_assets import load_manifest, pgt_root, STYLE_IDS, STYLES

items = load_manifest()
print("splits", Counter(it["split"] for it in items))
val = [it for it in items if it["split"] == "val"]
print("val", len(val), Counter(it["style"] for it in val))
for s in STYLES:
    p = pgt_root() / "orig_mean_fp16" / f"{STYLE_IDS[s]:02d}_{s}.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    ids = set(d["ids"])
    n_val = sum(1 for it in val if it["style"] == s and it["id"] in ids)
    print(s, "total", len(ids), "val_have", n_val, "/", sum(1 for it in val if it["style"] == s))
