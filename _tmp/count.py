"""Report dataset pool sizes and the split actually used for pseudo-GT."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
m = json.loads((ROOT / "data/wikiart_new_packed/manifest.json").read_text(encoding="utf-8"))
s = m["samples"]
print("total samples:", len(s))
print("schema:", m.get("schema"))
if isinstance(s, dict):
    s = list(s.values())
print("sample keys:", sorted(s[0].keys()))
split = {}
for rec in s:
    hit = "train"
    for k in ("split", "subset"):
        if k in rec:
            hit = str(rec[k])
    split[hit] = split.get(hit, 0) + 1
print("split keys:", split)
by_style = {}
for rec in s:
    st = rec.get("style") or rec.get("style_name")
    by_style[st] = by_style.get(st, 0) + 1
print("by style:", by_style)
