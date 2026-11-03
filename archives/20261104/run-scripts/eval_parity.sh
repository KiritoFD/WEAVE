#!/usr/bin/env bash
# §9.5 eval batch parity on a small stratified strict-reference subset (n=10).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
CKPT=/mnt/i/Latent_Style/runs/p0_diag/baseline_nohh600/ckpt/epoch_0012.pt
CFG=/mnt/i/Latent_Style/runs/p0_diag/baseline_nohh600/config.json
R=/mnt/i/Latent_Style/runs/p0_diag
for B in 2 32; do
  rm -rf "$R/parity_b$B"
  SUBSET=stratified SUBSET_N=10 REF_POOL=strict EVAL_BATCH=$B DINO_BATCH=16 \
    bash runs/eval_one_fast.sh "parity_b$B" "$CKPT" "$R" "$CFG" > "$R/parity_b$B.stdout" 2>&1
  echo "parity_b$B done rc=$?"
done
python3 - "$R" <<'PY'
import json, os, sys
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
    for k in ("dino_s", "clip_s", "dino_c", "lpips"):
        if k in a[block] and k in b[block]:
            d = abs(b[block][k] - a[block][k])
            assert d < 1e-4, (block, k, d)
pa = os.path.join(r, "parity_b2/images")
pb = os.path.join(r, "parity_b32/images")
mx = 0
n = 0
for fn in sorted(os.listdir(pa)):
    p2 = os.path.join(pa, fn)
    p32 = os.path.join(pb, fn)
    if not os.path.exists(p32) or not fn.lower().endswith((".png", ".jpg")):
        continue
    ia = np.asarray(Image.open(p2).convert("RGB"), dtype=np.int16)
    ib = np.asarray(Image.open(p32).convert("RGB"), dtype=np.int16)
    mx = max(mx, int(np.abs(ia - ib).max()))
    n += 1
print("PIXELS compared=%d max_abs_diff=%d (=%.6f/255)" % (n, mx, mx))
print("PARITY_OK" if mx <= 1 else "PARITY_FAIL")
PY
echo EVAL_PARITY_DONE
