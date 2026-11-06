import json
import sys
from collections import Counter

sys.path.insert(0, "/mnt/i/Latent_Style/WEAVE")
from tools.pseudo_gt.build_assets import load_manifest, STYLES

items = load_manifest()
print("n_items", len(items))
print("keys", sorted(items[0].keys()))
print("sample", json.dumps(items[0], ensure_ascii=False)[:400])
for s in STYLES:
    sub = [it for it in items if it["style"] == s]
    authors = Counter(it.get("author", "") for it in sub)
    print(s, len(sub), "authors", len(authors), authors.most_common(4))
