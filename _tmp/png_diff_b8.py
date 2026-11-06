import pathlib

import numpy as np
from PIL import Image

ref = pathlib.Path("/mnt/i/Latent_Style/pseudo_gt_24h/eval_render/S1_A_P5_final/images")
root = pathlib.Path("/mnt/i/Latent_Style/pseudo_gt_24h/bench_render")
names = sorted(p.name for p in ref.glob("*.png"))
for c in ["bench_b32", "bench_b64", "bench_b128"]:
    d = root / c / "images"
    vals = []
    frac_any = 0
    for name in names[:60]:
        a = np.asarray(Image.open(ref / name), dtype=np.int16)
        b = np.asarray(Image.open(d / name), dtype=np.int16)
        df = np.abs(a - b)
        vals.append(float(df.mean()))
        frac_any += int((df > 0).any())
    print(c, "mean_abs_u8", round(float(np.mean(vals)), 4),
          "max_abs_u8", int(max(np.abs(np.asarray(Image.open(ref / n), dtype=np.int16)
                                       - np.asarray(Image.open(d / n), dtype=np.int16)).max()
                               for n in names[:10])),
          "imgs_with_any_diff", frac_any, "/60")
