#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
for d in reverse_e1 reverse_e3 reverse_e8; do
  echo "=== $d ==="; ls -la "../pseudo_gt_24h/pairs/$d"
  python3 - <<PY
import json,pathlib
p=pathlib.Path('../pseudo_gt_24h/pairs/$d/manifest.json')
if p.exists():
    m=json.loads(p.read_text())
    print({k:v for k,v in m.items() if k!='records'})
    if 'records' in m: print('records', len(m['records']), 'first', m['records'][0].get('sample_id'))
else:
    print('NO MANIFEST')
PY
done
echo "=== POSITIVE MANIFEST SUMMARY ==="; python3 - <<'PY'
import json,pathlib
m=json.loads(pathlib.Path('../pseudo_gt_24h/pairs/positive/manifest.json').read_text())
print({k:v for k,v in m.items() if k!='records'}, 'n_records', len(m.get('records',[])))
PY
echo "=== REVERSE MANIFEST SUMMARY ==="; python3 - <<'PY'
import json,pathlib
m=json.loads(pathlib.Path('../pseudo_gt_24h/pairs/reverse/manifest.json').read_text())
print({k:v for k,v in m.items() if k!='records'}, 'n_records', len(m.get('records',[])))
PY
echo "=== SHARDS ==="; for d in positive reverse reverse_e1 reverse_e3; do echo "$d shards=$(ls -1 ../pseudo_gt_24h/pairs/$d/shard_*.pt 2>/dev/null | wc -l)"; done
echo "=== DISK ==="; df -h /mnt/i | tail -1
