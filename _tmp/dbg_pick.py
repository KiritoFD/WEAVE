import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

import paths
from tools.pseudo_gt.build_assets import STYLES, load_manifest, pgt_root
from tools.pseudo_gt.masks import E_SPECS, PATCH_SPECS, build_patch_layout
from tools.pseudo_gt.matching import DEVICE, match_rect


def main() -> None:
    items = load_manifest()
    dino = {}
    for s in STYLES:
        p = torch.load(pgt_root() / "dino_224" / f"{s}.pt", map_location="cpu", weights_only=False)
        dino[s] = {"ids": p["ids"], "index": {x: i for i, x in enumerate(p["ids"])}, "patches": p["patches"]}
        del p
    ret = json.loads((pgt_root() / "retrieval" / "top8.json").read_text(encoding="utf-8"))["styles"]
    specs = list(PATCH_SPECS) + list(E_SPECS)
    layouts = {n: build_patch_layout(n, seed=2024) for n in specs}
    sources = sorted([it for it in items if it["style"] == "Early_Renaissance" and it["split"] == "train"],
                     key=lambda it: it["id"])[:2]
    for it in sources:
        tgt = "Impressionism"
        cands = ret[it["style"]]["by_target"][tgt]["donor_ids"][ret[it["style"]]["source_ids"].index(it["id"])][:8]
        sp = F.normalize(dino[it["style"]]["patches"][dino[it["style"]]["index"][it["id"]]].float(), dim=-1).to(DEVICE)
        print("SRC", it["id"][:45], "cands", len(cands), flush=True)
        for d in cands:
            dp = F.normalize(dino[tgt]["patches"][dino[tgt]["index"][d]].float(), dim=-1).to(DEVICE)
            ok_specs, mins = 0, {}
            for name, rects in layouts.items():
                qs = []
                for r in rects:
                    m = match_rect(sp, dp, r, min_q=-2.0)
                    qs.append(-2.0 if m is None else m.q)
                mins[name] = round(min(qs), 3)
                if min(qs) >= 0.30:
                    ok_specs += 1
            print("  donor", d[:40], "ok", ok_specs, "/", len(layouts), "mins", mins, flush=True)


if __name__ == "__main__":
    main()
