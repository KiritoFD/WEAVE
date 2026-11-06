import csv
from pathlib import Path

root = Path("/mnt/i/Latent_Style/pseudo_gt_24h/bench_render")


def load(cell, fname="dino_metrics.csv"):
    with open(root / cell / fname, newline="", encoding="utf-8") as fh:
        return {(r["src_image"], r["tgt_style"]): r for r in csv.DictReader(fh)}


a = load("b64_t8")
b = load("b64_m128_t8")
keys = [k for k in a if k in b]
print("pairs", len(a), len(keys))
for col in ("dino_style_similarity", "dino_content_similarity"):
    diffs = [abs(float(a[k][col]) - float(b[k][col])) for k in keys]
    print(f"{col}: max_abs_diff={max(diffs):.3e} mean={sum(diffs)/len(diffs):.3e}")

ma = load("b64_t8", "metrics.csv")
mb = load("b64_m128_t8", "metrics.csv")
keys2 = [k for k in ma if k in mb]
for col in ("clip_style", "lpips"):
    diffs = [abs(float(ma[k][col]) - float(mb[k][col])) for k in keys2]
    print(f"{col}: max_abs_diff={max(diffs):.3e}")
