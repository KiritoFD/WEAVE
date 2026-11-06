import hashlib
import pathlib
import numpy as np
from PIL import Image

root = pathlib.Path("/mnt/i/Latent_Style/pseudo_gt_24h/bench_render")
ref_dir = root / "bench_b32" / "images"
for c in ["bench_b64", "bench_b96", "bench_b128"]:
    d = root / c / "images"
    same = diff = 0
    diffs = []
    names = sorted(p.name for p in ref_dir.glob("*.png"))
    for name in names[:30]:
        a = np.asarray(Image.open(ref_dir / name), dtype=np.int16)
        b = np.asarray(Image.open(d / name), dtype=np.int16)
        diffs.append(float(np.abs(a - b).mean()))
    for name in names:
        ha = hashlib.md5(open(ref_dir / name, "rb").read()).hexdigest()
        hb = hashlib.md5(open(d / name, "rb").read()).hexdigest()
        if ha == hb:
            same += 1
        else:
            diff += 1
    print(c, "same", same, "diff", diff, "mean_abs_u8_diff_30imgs", round(float(np.mean(diffs)), 4),
          "max", round(float(np.max(diffs)), 4))
