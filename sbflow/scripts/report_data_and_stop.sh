#!/usr/bin/env bash
set -uo pipefail
pkill -f "sbflow\." || true
sleep 2
cd /mnt/i/Latent_Style/WEAVE
echo "=== training images ==="
for S in Impressionism Ukiyo_e; do
  N=$(find "datasets/distinct5_512/train/$S" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  echo "  distinct5_512/train/$S: $N"
done
echo "=== test sources (frozen) ==="
for S in Impressionism Ukiyo_e; do
  N=$(find "data/test/$S" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  echo "  data/test/$S: $N"
done
echo "=== latent caches ==="
ls -lh runs/blog_sbflow/cache/
python3 - <<'PY'
import torch
for s in ("Impressionism","Ukiyo_e"):
    d = torch.load(f"runs/blog_sbflow/cache/{s}.pt", map_location="cpu", weights_only=False)
    print(f"  {s}: latent {tuple(d['latent'].shape)} {d['latent'].dtype}, thumb {tuple(d['thumb'].shape)} {d['thumb'].dtype}, n_ids={len(d['ids'])}")
PY
echo "=== frozen task manifest ==="
python3 - <<'PY'
import json
raw = json.load(open("eval_protocol/tasks_750.json"))
t = raw["tasks"]
pair = [x for x in t if x["source_style"] in ("Impressionism","Ukiyo_e")
        and x["target_style"] in ("Impressionism","Ukiyo_e")
        and x["source_style"] != x["target_style"]]
print(f"  tasks_750 total={len(t)}  Impressionism<->Ukiyo_e cross={len(pair)}")
print(f"  distinct source images in that pair: {len({x['source_filename'] for x in pair})}")
PY
