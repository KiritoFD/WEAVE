#!/usr/bin/env bash
# P0.2 Capacity / overfitting diagnostic.
# Train the baseline recipe (no new modules) on a deliberately tiny content set:
#   cap32  -> first 32 latents per style (160 total)
#   cap128 -> first 128 latents per style (640 total)
# 5000 optimiser updates each, starting from the same epoch_0010 checkpoint.
# Then report (a) fit on the training images, (b) held-out 150 same-style.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/p0_diag
mkdir -p "$R"

# --- extract the first 32 images per style from the packed uint8 buffer ---
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np
from PIL import Image
root = Path('/mnt/i/Latent_Style/WEAVE/data/train_d5')
idx = json.load(open(root / 'images_index.json'))
recs = idx['records']; H = idx['height']; W = idx['width']
buf = np.memmap(root / 'images_hwc_uint8.bin', dtype=np.uint8, mode='r',
                shape=(len(recs), H, W, 3))
styles = list(dict.fromkeys(r['style'] for r in recs))
per_style = 32
out = Path('/mnt/i/Latent_Style/runs/p0_diag/fit32')
kept = {}
for s in styles:
    rows = [i for i, r in enumerate(recs) if r['style'] == s][:per_style]
    kept[s] = [recs[i]['file'] for i in rows]
    d = out / s; d.mkdir(parents=True, exist_ok=True)
    for i in rows:
        stem = Path(recs[i]['file']).stem
        Image.fromarray(np.asarray(buf[i])).save(d / (stem + '.jpg'), quality=95)
print('[p0] fit32 images:', sum(len(v) for v in kept.values()),
      '| first/last ER:', kept[styles[0]][0], kept[styles[0]][-1])
# sanity: latent stem order must match the packed image order
lat = sorted(p.stem for p in (Path('/mnt/i/Latent_Style/WEAVE/data/train') / styles[0]).glob('*.pt'))[:per_style]
assert lat == sorted(Path(v).stem for v in kept[styles[0]]), 'image/latent order mismatch'
print('[p0] fit32 order matches latent order')
PY

train_and_eval () {
  n="$1"; tag="cap$n"; steps=5000; b=14
  dir="$R/$tag"; mkdir -p "$dir"
  vlm=$(python3 -c "print($steps*$b/($n*5))")
  python3 - "$dir/config.json" "$tag" "$n" "$vlm" <<'PY'
import json, sys
cfg, tag, n, vlm = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/p0_diag/%s/ckpt" % tag},
     "training": {"batch_size": 14, "num_epochs": 11, "seed": 0, "log_interval": 10,
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "save_interval": 1},
     "data": {"packed_limit_per_style": n, "virtual_length_multiplier": vlm}}
json.dump(d, open(cfg, 'w'), indent=2)
PY
  echo "===== $(date '+%F %T') TRAIN $tag limit=$n/style steps=$steps vlm=$vlm =====" | tee -a "$R/capacity.log"
  timeout 3600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
  echo "[train rc=$?] $tag" | tee -a "$R/capacity.log"
  grep -a "Epoch 11/11" "$dir/train.log" | tail -1 | tee -a "$R/capacity.log"
  ck="$dir/ckpt/epoch_0011.pt"
  if [ -f "$ck" ]; then
    bash /mnt/i/Latent_Style/runs/eval_one.sh "${tag}_fit" "$ck" "$R" "$dir/config.json" "" \
      /mnt/i/Latent_Style/runs/p0_diag/fit32 2>&1 | tail -3 | tee -a "$R/capacity.log"
    bash /mnt/i/Latent_Style/runs/eval_one.sh "${tag}_heldout" "$ck" "$R" "$dir/config.json" 2>&1 | tail -3 | tee -a "$R/capacity.log"
  else
    echo "[NO CKPT] $tag" | tee -a "$R/capacity.log"
  fi
}

train_and_eval 32
train_and_eval 128
echo P0_CAPACITY_DONE | tee -a "$R/capacity.log"
