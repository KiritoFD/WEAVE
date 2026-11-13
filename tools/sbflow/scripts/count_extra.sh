#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== wikiart_random20_512 counts ==="
for SPLIT in train test; do
  TOT=0; LINE="  random20/$SPLIT:"
  for S in Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e; do
    D="datasets/wikiart_random20_512/wikiart_random20_512/images/$SPLIT/$S"
    if [ -d "$D" ]; then N=$(find "$D" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l); else N=0; fi
    LINE="$LINE $S=$N"; TOT=$((TOT+N))
  done
  echo "$LINE | total=$TOT"
done
echo
echo "=== D5 pseudo-GT latent caches (what V8/V9 loaders actually read) ==="
python3 - <<'PY'
import glob, torch, os
files = sorted(glob.glob("/mnt/i/Latent_Style/**/pseudo_gt_24h/orig_mean_fp16/*.pt", recursive=True))
if not files:
    files = sorted(glob.glob("/mnt/i/Latent_Style/**/orig_mean_fp16/*.pt", recursive=True))
tot = 0
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        for k in ("latent", "z", "latents", "data"):
            if k in d:
                d = d[k]; break
    try:
        n = int(d.shape[0]); shape = tuple(d.shape[1:])
    except Exception:
        n, shape = -1, "?"
    tot += max(n, 0)
    print(f"  {os.path.basename(f):32s} n={n:6d} per-sample={shape} dtype={getattr(d,'dtype','?')}")
print(f"  TOTAL = {tot}")
PY
