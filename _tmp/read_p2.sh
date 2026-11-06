#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "--- idset check ---"
python3 - <<'PY'
import json
from pathlib import Path
R=Path('/mnt/i/Latent_Style/pseudo_gt_24h/pairs')
def ids(d):
    m=json.loads((R/d/'manifest.json').read_text())
    return {r['sample_id'] for r in m['records']}, len(m['records'])
a=ids('base'); b=ids('positive'); c=ids('reverse'); e=ids('reverse_e8')
print('base',a[1],'positive',b[1],'reverse',c[1],'e8',e[1])
print('base&positive',len(a[0]&b[0]),'base&reverse',len(a[0]&c[0]),'pos&rev',len(b[0]&c[0]))
print('sample_pos', list(b[0])[:2])
PY
echo "--- p2 json (truncated per arm) ---"
python3 - <<'PY'
import json
d=json.load(open('/mnt/i/Latent_Style/pseudo_gt_24h/p2_reach.json'))
for k,v in d.items():
    print(k, json.dumps({kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk!='band_share'}, ensure_ascii=False))
    print('   band_share', {kk:round(vv,4) for kk,vv in v['band_share'].items()})
PY
