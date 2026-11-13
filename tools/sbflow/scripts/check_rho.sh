#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
for a in ("d1","d2","e1","e2"):
    p = Path(f"runs/blog_sbflow/v11/{a}/train_summary.json")
    if not p.exists():
        print(f"  {a}: no summary yet"); continue
    s = json.loads(p.read_text())
    h = s.get("loss_history", [])
    def first(k):
        for x in h:
            if x.get(k) is not None:
                return x[k]
        return None
    print(f"  {a}: updates={s.get('updates')} sec/upd={s.get('sec_per_update')} "
          f"peak={s.get('peak_vram_mib')} exposures={s.get('endpoint_exposures')}")
    print(f"      dist_grad_norm={first('dist_grad_norm')} "
          f"bridge_grad_norm={first('bridge_grad_norm')} "
          f"rho={first('rho_swd_over_bridge')}")
    print(f"      first hist entry keys: {sorted(h[0].keys()) if h else '-'}")
PY
