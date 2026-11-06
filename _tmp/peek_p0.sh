#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
d = json.load(open("runs/p2/pgt_p0_audit.json"))
print(sorted(d.keys()))
for k in d:
    v = d[k]
    if isinstance(v, dict) and len(json.dumps(v)) > 1500:
        print("==", k, json.dumps({kk: v[kk] for kk in list(v)[:8]}, ensure_ascii=False)[:1200])
    else:
        print("==", k, json.dumps(v, ensure_ascii=False)[:1200])
PY
