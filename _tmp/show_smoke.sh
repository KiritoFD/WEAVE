#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
d = json.load(open('/tmp/pgt_q0_smoke.json'))
print('n_source', d.split if False else d['n_source'])
# error rows are not in summarise output; check raw rows? summarise drops them.
print(json.dumps({k: v for k, v in list(d['configs'].items())[:3]}, ensure_ascii=False)[:800])
PY
echo EXIT=$?
