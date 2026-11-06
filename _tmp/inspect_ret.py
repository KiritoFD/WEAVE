import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import paths
from tools.pseudo_gt.build_assets import STYLES, load_manifest, pgt_root

items = load_manifest()
ret = json.loads((pgt_root() / "retrieval" / "top8.json").read_text(encoding="utf-8"))["styles"]
print("manifest", len(items))
for s in STYLES:
    print(s, "ret src", len(ret[s]["source_ids"]), "row0", len(ret[s]["donor_ids"][0]))
ids_by_style = {}
for s in STYLES:
    p = torch.load
    import torch
    pay = torch.load(pgt_root() / "dino_224" / f"{s}.pt", map_location="cpu", weights_only=False)
    ids_by_style[s] = set(pay["ids"])
    print(s, "dino ids", len(pay["ids"]))
    del pay
src_index = {}
for s in STYLES:
    src_index[s] = {v: i for i, v in enumerate(ret[s]["source_ids"])}
for s in STYLES:
    missing = 0
    cands_counts = []
    for it in [x for x in items if x["style"] == s and x["split"] == "train"][:200]:
        i = src_index[s].get(it["id"])
        if i is None:
            missing += 1
            continue
        tgt = STYLES[(STYLES.index(s) + 1) % 5]
        raw = ret[s]["donor_ids"][i]
        ok = [c for c in raw if c in ids_by_style[tgt]]
        cands_counts.append(len(ok))
    print(s, "missing_source", missing, "ok_counts sample", cands_counts[:10])
