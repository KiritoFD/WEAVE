#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
p='/mnt/i/Latent_Style/pseudo_gt_24h/eval_render/S2_E7_final/v4_eval750_rows.json'
d=json.load(open(p))
print(type(d))
if isinstance(d,dict): print(list(d.keys())[:12])
rows = d if isinstance(d,list) else d.get('rows', d)
print('n_rows', len(rows))
print('row0', json.dumps(rows[0], ensure_ascii=False)[:900])
PY
echo "--- e3 build ---"; tail -n 4 runs/v4/build_reverse_e3.log 2>/dev/null | cut -c1-200
ls -la ../pseudo_gt_24h/pairs/reverse_e3 2>/dev/null | tail -3
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
