"""Compare two eval dirs produced with different EVAL_BATCH values (§9.5)."""
import json
import os
import sys

import numpy as np
from PIL import Image

r = sys.argv[1]
a = json.load(open(os.path.join(r, "parity_b2/results_split.json")))
b = json.load(open(os.path.join(r, "parity_b32/results_split.json")))
print("counts b2=%s b32=%s" % (a["counts"], b["counts"]))
for block in ("main_cross_style", "identity_same_style"):
    for k in ("dino_s", "clip_s", "dino_c", "lpips"):
        if k in a[block] and k in b[block]:
            print("METRIC %-18s %-8s delta=%+.8f" % (block, k, b[block][k] - a[block][k]))

pa = os.path.join(r, "parity_b2/images")
pb = os.path.join(r, "parity_b32/images")
mx = 0
tot = 0
n = 0
for fn in sorted(os.listdir(pa)):
    if not fn.lower().endswith((".png", ".jpg")):
        continue
    p32 = os.path.join(pb, fn)
    if not os.path.exists(p32):
        continue
    ia = np.asarray(Image.open(os.path.join(pa, fn)).convert("RGB"), dtype=np.int16)
    ib = np.asarray(Image.open(p32).convert("RGB"), dtype=np.int16)
    d = np.abs(ia - ib)
    mx = max(mx, int(d.max()))
    tot += int(d.sum())
    n += 1
print("PIXELS compared=%d max_abs_diff=%d mean_abs_diff=%.4f" % (n, mx, tot / max(n * ia.size, 1)))
print("PARITY_%s" % ("OK" if mx <= 1 else "FAIL"))

