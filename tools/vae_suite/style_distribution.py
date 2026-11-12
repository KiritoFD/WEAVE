"""Per-style image distribution in the WikiArt training pool."""
import sys, json, collections
sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
from pathlib import Path
import numpy as np
from tools.vae_suite.dataset import RawMemmapWikiArtDataset

MAN = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
ds = RawMemmapWikiArtDataset(MAN, is_training=True)
man = json.loads(MAN.read_text())
id2name = {v: k for k, v in man["style_to_id"].items()}

sa = collections.defaultdict(lambda: collections.defaultdict(int))
for pos, raw in enumerate(ds.indices):
    r = ds.records_by_idx[raw]
    sa[r["style_id"]][r.get("artist_id", -1)] += 1

rows = []
for s, am in sa.items():
    n = sum(am.values())
    nart = len([a for a, v in am.items() if a >= 0 and v])
    rows.append((s, id2name.get(s, "?"), n, nart))
rows.sort(key=lambda x: -x[2])
N = sum(r[2] for r in rows)

print(f"TOTAL {N} images, {len(rows)} styles\n")
print("=== cumulative coverage by threshold ===")
for th in (2000, 1000, 500, 200, 100, 50, 20, 10):
    keep = [r for r in rows if r[2] >= th]
    img = sum(r[2] for r in keep)
    print(f"  >= {th:5d}: {len(keep):3d} styles | {img:6d} images ({img/N*100:5.1f}%)")

print("\n=== full table (sorted desc) ===")
hdr = f"{'rank':>4} {'id':>4} {'name':<34} {'imgs':>6} {'artists':>7} {'cum%':>7}"
print(hdr)
print("-" * len(hdr))
cum = 0
for i, (s, nm, n, na) in enumerate(rows, 1):
    cum += n
    print(f"{i:>4} {s:>4} {nm[:34]:<34} {n:>6} {na:>7} {cum/N*100:>6.1f}%")
