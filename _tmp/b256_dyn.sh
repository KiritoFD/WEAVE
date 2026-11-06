#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json,glob,os
for cell in ("S0_base_b256","S1_positive_b256","S2_reverse_b256"):
    f=f"runs/pgt/{cell}/internal_dynamics_summary.json"
    if not os.path.exists(f):
        print(cell,"no summary"); continue
    d=json.load(open(f))
    r=d.get("rho_series") or []; g=d.get("gate_mean_series") or []
    print(cell,"fired=%s ep=%s step=%s reason=%s"%(d.get("fired"),d.get("fired_epoch"),d.get("global_step"),d.get("stop_reason")))
    if r: print("   last_rho=%s prev_rho=%s ratio=%.3f gate_delta=%+.4f"%(r[-1],r[-2] if len(r)>1 else -1,(r[-1]/r[-2]) if len(r)>1 else float('nan'),(g[-1]-g[-2]) if len(g)>1 else float('nan')))
PY
tail -n 3 runs/pgt/PROGRESS.md
