#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== p3 uncond summary ==="
cat runs/blog_sbflow/v11/p3_score/uncond_summary.md 2>/dev/null || echo "  missing"
echo
echo "=== p3 train summary ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/blog_sbflow/v11/p3/train_summary.json")
if p.exists():
    s = json.loads(p.read_text())
    print(f"  updates={s.get('updates')} sec/upd={s.get('sec_per_update')} "
          f"peak={s.get('peak_vram_mib')} exposures={s.get('endpoint_exposures')}")
    print(f"  pairs={len(s.get('unordered_pairs', []))} styles={s.get('styles')}")
    print(f"  counts={s.get('counts')}")
PY
