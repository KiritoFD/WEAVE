#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
D=$(ls -d /mnt/i/Latent_Style/eval_assets/p2_render/*/ 2>/dev/null | head -3)
for d in $D; do echo "-- $d"; ls "$d" | head -5; head -n 2 "$d/metrics.csv" 2>/dev/null; done
echo "--- render dirs"
ls -1 /mnt/i/Latent_Style/eval_assets/p2_render 2>/dev/null
echo "--- p2 json keys"
python3 - <<'PY'
import json, glob
f = sorted(glob.glob("runs/p2/pgt_*.json"))
print(f[:5])
if f:
    d = json.load(open(f[0]))
    print(sorted(d.keys())[:40])
PY
