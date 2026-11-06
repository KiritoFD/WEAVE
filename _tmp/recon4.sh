#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "=== PAIRS ==="; for d in ../pseudo_gt_24h/pairs/*; do echo "$d : $(ls -1 "$d" 2>/dev/null | wc -l) files"; done
echo "=== PAIRS DETAIL ==="; ls -la ../pseudo_gt_24h/pairs/
echo "=== E_SPECS ==="; python3 - <<'PY'
from tools.pseudo_gt.masks import E_SPECS
import json
for k,v in E_SPECS.items():
    print(k, json.dumps(v, ensure_ascii=False)[:220])
PY
echo "=== S1 PAIRS SAMPLE ==="; python3 - <<'PY'
from pathlib import Path
import json
root=Path('../pseudo_gt_24h/pairs/positive')
for p in sorted(root.iterdir())[:4]:
    print(p.name, p.stat().st_size)
    if p.suffix=='.json':
        print(p.read_text()[:400])
PY
echo "=== MD5 KEY FILES ==="; md5sum tools/pseudo_gt/build_train_cache.py tools/pseudo_gt/masks.py tools/pseudo_gt/construct.py tools/pseudo_gt/pgt_render.py run.py configs/v4/S3_E3.json configs/v4/S4a_E1.json 2>&1
