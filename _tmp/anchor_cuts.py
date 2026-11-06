import csv
from collections import defaultdict

base = "/mnt/i/Latent_Style/WEAVE/runs/12h/anchors/base_ep10_750_512"
m = list(csv.DictReader(open(f"{base}/metrics.csv", newline="", encoding="utf-8")))
d = list(csv.DictReader(open(f"{base}/dino_metrics.csv", newline="", encoding="utf-8")))
print("cols_metrics", list(m[0])[:14])
print("cols_dino", list(d[0]))
print("rows", len(m), len(d))

aggs = defaultdict(lambda: defaultdict(list))
for r in d:
    cross = r["src_style"] != r["tgt_style"]
    g = "cross600" if cross else "same150"
    for grp in (g, "all750"):
        aggs[grp]["dino_s"].append(float(r["dino_style_similarity"]))
        aggs[grp]["dino_c"].append(float(r["dino_content_similarity"]))
for r in m:
    cross = r["src_style"] != r["tgt_style"]
    g = "cross600" if cross else "same150"
    for grp in (g, "all750"):
        aggs[grp]["clip_s"].append(float(r["clip_style"]))
        aggs[grp]["lpips"].append(float(r["lpips"]))
for g, kv in aggs.items():
    print(g, {k: round(sum(v) / len(v), 5) for k, v in kv.items()}, "n=", len(next(iter(kv.values()))))
