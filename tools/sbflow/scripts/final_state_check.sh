#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== tmux ==="
tmux ls 2>&1
echo
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo
echo "=== compute processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "  none"
echo
echo "=== pipeline markers ==="
tail -n 3 runs/v6_24h/pass3.log 2>/dev/null || echo "  no pass3 log"
echo "  ---"
tail -n 2 runs/v6_24h/postqueue.log 2>/dev/null || echo "  no postq log"
echo
echo "=== arms done ==="
python3 -c "
import json
from pathlib import Path
s = json.loads(Path('runs/v6_24h/state.json').read_text())
done = [k for k, v in s.items() if v.get('status') == 'done']
other = {k: v.get('status') for k, v in s.items() if v.get('status') != 'done'}
print(f'  {len(done)} done')
if other: print('  non-done:', other)
else: print('  no failed/incomplete/blocked arms')
"
