#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
p='runs/12h/anchors/base_ep10_750_512/results_split.json'
d=json.load(open(p))
print('top keys', list(d.keys())[:10])
def show(name, node):
    if not isinstance(node, dict): return
    ks=[k for k in ('n','DINO_S','DINO_C','CLIP_S','LPIPS','CSD_dT','CSD_dS','CSD_dM','dino_s','dino_c','clip_s','lpips') if k in node]
    print(name, {k:node[k] for k in ks})
for k,v in d.items():
    if isinstance(v, dict): show(k,v)
PY
echo "--- progress ---"; tail -n 3 runs/gt/PROGRESS.md
echo "--- build ---"; tail -n 4 runs/v4/build_reverse_e3.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
