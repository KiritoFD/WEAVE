"""Exposure / sampling analysis under different style-count thresholds."""
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

print("=== tail of table (rank 56-136) ===")
for i, (s, nm, n, na) in enumerate(rows[55:], 56):
    print(f"{i:>4} {s:>4} {nm[:36]:<36} {n:>6} {na:>7}")

TOT_MICRO = 2740 * 8   # total microbatches in the 12h plan

print("\n=== exposure under style-balanced sampling (12h plan, 2740 updates x 8 = 21,920 microbatches) ===")
print(f"{'threshold':>10}{'styles':>8}{'images':>9}{'img%':>7}"
      f"{'head_exp':>10}{'tail_exp':>10}{'max/min':>9}{'median_exp':>11}")
print("-" * 74)
for th in (0, 100, 200, 500, 1000, 2000, 3000):
    keep = [r for r in rows if r[2] >= th]
    if not keep:
        continue
    ns = np.array([r[2] for r in keep], dtype=float)
    imgs = int(ns.sum())
    per_img = TOT_MICRO * 3.0 / (len(keep) * ns)     # expected exposures per image
    print(f"{th:>10}{len(keep):>8}{imgs:>9}{imgs/100207*100:>6.1f}%"
          f"{per_img.min():>10.2f}{per_img.max():>10.2f}"
          f"{per_img.max()/per_img.min():>9.1f}{np.median(per_img):>11.2f}")

print("\n=== per-epoch image count if we drop below threshold ===")
for th in (1000, 2000):
    keep = [r for r in rows if r[2] >= th]
    imgs = sum(r[2] for r in keep)
    print(f"  >= {th}: {len(keep)} styles, {imgs} images | "
          f"1 epoch = {imgs} images = {imgs/48:.0f} updates = {imgs/48*11.85/3600:.2f} h (S1 speed)")
