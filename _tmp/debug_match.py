import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "pseudo_gt"))

import torch
import torch.nn.functional as F
import paths
from masks import build_patch_layout
import matching

root = paths.asset_root() / "pseudo_gt_24h"
d = {}
for style in ["Early_Renaissance", "Impressionism"]:
    p = torch.load(root / "dino_224" / f"{style}.pt", map_location="cpu", weights_only=False)
    d[style] = {"cls": F.normalize(p["cls"].float(), dim=-1), "p": F.normalize(p["patches"].float(), dim=-1), "ids": p["ids"]}

src_style, tgt_style = "Early_Renaissance", "Impressionism"
si = d[src_style]["ids"][0]
src_pat = d[src_style]["p"][d[src_style]["ids"].index(si)][None]
best = None
for di in range(5):
    tid = d[tgt_style]["ids"][di]
    don_pat = d[tgt_style]["p"][d[tgt_style]["ids"].index(tid)][None]
    print("donor", tid[:40])
    for spec in ["P1", "P3", "P5", "P7", "P9"]:
        rects = build_patch_layout(spec, seed=7)
        print(" spec", spec, "rects", [(r.x, r.y, r.w, r.h) for r in rects][:3])
        for r in rects:
            sy, sx = matching._rect_tokens(r)
            m = matching.match_rect(src_pat[0], don_pat[0], r, min_q=0.30)
            print("   rect", (r.x, r.y, r.w, r.h), "tokens", (sy.start, sy.stop, sx.start, sx.stop),
                  "->", None if m is None else (round(m.q, 3), round(m.mutual_ratio, 3)))
