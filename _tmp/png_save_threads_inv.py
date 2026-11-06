import hashlib
import pathlib

root = pathlib.Path("/mnt/i/Latent_Style/pseudo_gt_24h/bench_render")
a = root / "bench_b64" / "images"
b = root / "b64_t8" / "images"
names = sorted(p.name for p in a.glob("*.png"))
same = diff = 0
for n in names:
    ha = hashlib.md5((a / n).read_bytes()).hexdigest()
    hb = hashlib.md5((b / n).read_bytes()).hexdigest()
    same += int(ha == hb)
    diff += int(ha != hb)
print("b64_vs_b64_t8 same", same, "diff", diff, "total", len(names))
