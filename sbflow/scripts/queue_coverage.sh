#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== arms the running queue has started (from queue.log) ==="
grep -a "^\[queue\] === " runs/v6_24h/queue.log | sed 's/\[queue\] === //' | cut -d' ' -f1 | nl
echo
echo "=== arms in the current arms.py table (21) ==="
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 -c "
from tools.v6_24h.arms import ordered
ids=[a['id'] for a in ordered()]
print(len(ids), 'arms')
for i,x in enumerate(ids,1): print(f'  {i:2d} {x}')
"
echo
echo "=== done per state.json ==="
python3 -c "
import json
from pathlib import Path
s=json.loads(Path('runs/v6_24h/state.json').read_text())
d=[k for k,v in s.items() if v.get('status')=='done']
print(len(d),'done:',', '.join(sorted(d)))
"
