#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
d=json.load(open('runs/p2/v4_pattern_report.json'))
print('top-level keys:', list(d.keys()))
def show(obj, depth=0):
    if isinstance(obj,dict):
        for k,v in list(obj.items())[:12]:
            if isinstance(v,(dict,list)):
                print('  '*depth+str(k), type(v).__name__, len(v))
                if depth<2: show(v, depth+1)
            else:
                print('  '*depth+str(k),'=',v)
    elif isinstance(obj,list) and obj and isinstance(obj[0],dict):
        print('  '*depth+'first:', {k:obj[0][k] for k in list(obj[0])[:8]})
show(d)
PY
