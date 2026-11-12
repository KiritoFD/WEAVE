#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== evaluator_v2_all_results.json ==="
if [ -f runs/v6/evaluator_v2_all_results.json ]; then
  du -h runs/v6/evaluator_v2_all_results.json
  python3 -c "
import json
from pathlib import Path
d = json.loads(Path('runs/v6/evaluator_v2_all_results.json').read_text())
print('  top keys:', list(d.keys())[:10] if isinstance(d, dict) else f'list len {len(d)}')
if isinstance(d, dict):
    for k in list(d.keys())[:3]:
        v = d[k]
        print(f'    {k}: {type(v).__name__}', list(v.keys())[:8] if isinstance(v, dict) else '')
"
else
  echo "  MISSING"
fi
echo
echo "=== _baseline_stage ==="
ls -la _baseline_stage/ 2>/dev/null || echo "  no _baseline_stage on host"
echo
echo "=== docs/baseline/images (extracted?) ==="
ls -d docs/baseline/images/* 2>/dev/null | head -6 || echo "  not extracted"
