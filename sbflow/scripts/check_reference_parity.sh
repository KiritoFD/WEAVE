#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== the file the baseline ranking was computed from ==="
ls -la runs/v6/evaluator_v2_all_results.json 2>/dev/null || echo "  MISSING"
echo
echo "=== references_v1.json structure ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("eval_protocol/references_v1.json")
if not p.exists():
    print("  missing"); raise SystemExit
d = json.loads(p.read_text())
print("  top keys:", list(d.keys())[:8])
for k, v in list(d.items())[:6]:
    if isinstance(v, dict):
        print(f"    {k}: dict keys={list(v.keys())[:6]}")
    elif isinstance(v, list):
        print(f"    {k}: list len={len(v)}  sample={v[0] if v else None}")
    else:
        print(f"    {k}: {v}")
PY
echo
echo "=== csd_proto.pt (what my eval_tiers uses) ==="
python3 - <<'PY'
import torch
from pathlib import Path
p = Path("/mnt/i/Latent_Style/pseudo_gt_24h/csd_proto.pt")
if p.exists():
    d = torch.load(p, map_location="cpu", weights_only=False)
    print("  type:", type(d).__name__)
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            except Exception:
                print(f"    {k}: {v}")
    else:
        print("  shape:", tuple(d.shape))
PY
echo
echo "=== how does eval_tiers build the prototypes? ==="
grep -n "def csd_prototypes" -A 18 tools/v5/route_c.py 2>/dev/null | head -24
