#!/usr/bin/env bash
# Verify the new generation path is pixel-equivalent to the reference eval_test600 path,
# and that raw latent dumping works.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
CK=runs/blog_sbflow/Mid8M_noOT/model_step1000.pt

echo "=== reference path (eval_test600 generate) ==="
rm -rf runs/blog_sbflow/_parity_ref
python3 -u -m sbflow.eval_test600 generate --repo-root . --ckpt $CK \
  --style-a Impressionism --style-b Ukiyo_e --output runs/blog_sbflow/_parity_ref \
  --weights raw --method heun --steps 4 --batch 2

echo "=== new path (sbflow.generate, images + raw latents) ==="
rm -rf runs/blog_sbflow/_parity_new runs/blog_sbflow/_parity_lat
python3 -u -m sbflow.generate --repo-root . --ckpt $CK \
  --style-a Impressionism --style-b Ukiyo_e --out runs/blog_sbflow/_parity_new \
  --steps 4 --method heun --batch 2 --workers 4 \
  --save-images --dump-latents runs/blog_sbflow/_parity_lat

echo "=== compare ==="
python3 -u - <<'PY'
import json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image

ref = Path("runs/blog_sbflow/_parity_ref")
new = Path("runs/blog_sbflow/_parity_new/Mid8M_noOT_model_step1000")
lat = Path("runs/blog_sbflow/_parity_lat/Mid8M_noOT_model_step1000")

ref_man = json.loads((ref / "manifest.json").read_text())["tasks"]
new_man = json.loads((new / "manifest.json").read_text())["tasks"]
assert [t["task_id"] for t in ref_man] == [t["task_id"] for t in new_man], "task order differs"

diffs, maxd, mism = [], 0, 0
for i, (a, b) in enumerate(zip(ref_man, new_man)):
    ia = np.asarray(Image.open(a["generated"]).convert("RGB"), dtype=np.int16)
    ib = np.asarray(Image.open(new / b["image"]).convert("RGB"), dtype=np.int16)
    assert ib.shape == ia.shape, (b["image"], ia.shape, ib.shape)
    d = np.abs(ia - ib)
    diffs.append(float(d.mean()))
    maxd = max(maxd, int(d.max()))
    if d.max() > 0:
        mism += 1
print(f"images compared: {len(diffs)}")
print(f"mean abs pixel diff: {np.mean(diffs):.4f} (max over images {max(diffs):.4f})")
print(f"max abs pixel diff: {maxd}")
print(f"images with any differing pixel: {mism}/{len(diffs)}")

files = sorted(lat.glob("batch_*.npy"))
z = np.concatenate([np.load(f) for f in files], 0)
print(f"dumped latents: {len(files)} files, array {z.shape} {z.dtype}, finite={np.isfinite(z).all()}")
print(f"latent mean={z.mean():.4f} std={z.std():.4f} min={z.min():.3f} max={z.max():.3f}")
print("PARITY", "PASS (bit-exact)" if maxd == 0 else f"DIFFERS by {maxd} levels")
PY
