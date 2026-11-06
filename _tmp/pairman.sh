#!/usr/bin/env bash
cd /mnt/i/Latent_Style/pseudo_gt_24h/pairs || exit 1
ls -la */manifest.json | awk '{print $5, $9}'
python3 - <<'PY'
import json, os
for p in ("base", "positive", "reverse"):
    f = os.path.join(p, "manifest.json")
    try:
        d = json.load(open(f))
    except Exception as e:
        print(p, "ERR", e); continue
    if isinstance(d, dict):
        print(p, {k: (v if not isinstance(v, list) else len(v)) for k, v in d.items() if not isinstance(v, dict)})
        for k, v in d.items():
            if isinstance(v, dict):
                print("   ", k, {kk: (vv if not isinstance(vv, list) else len(vv)) for kk, vv in v.items()})
PY
