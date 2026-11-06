#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "== dynamics =="
python3 - <<'PY'
import json,glob,os
for f in sorted(glob.glob("runs/pgt/*/internal_dynamics_summary.json")):
    d=json.load(open(f))
    cell=os.path.basename(os.path.dirname(f))
    r=d.get("rho_series") or []; g=d.get("gate_mean_series") or []
    print(cell, "armed=%s fired=%s fired_ep=%s step=%s reason=%s"%(d.get("armed"),d.get("fired"),d.get("fired_epoch"),d.get("global_step"),d.get("stop_reason")))
    print("   rho=%s"%["%.4f"%x for x in r])
    print("   gate=%s"%["%.4f"%x for x in g])
PY
echo "== 600 block =="
python3 - <<'PY'
import json,glob,os
for f in sorted(glob.glob("runs/p2/pgt_S*.json")):
    d=json.load(open(f)); cell=os.path.basename(f)[4:-5]
    print("###",cell)
    for k in ("600_cross_style","600","cross_style"):
        if k in d: blk=d[k]; break
    else:
        blk=None
    if blk is None:
        print("  keys:",list(d.keys())[:12]); continue
    print("  ",json.dumps(blk,ensure_ascii=False)[:900])
PY
echo "== files =="
ls -l runs/p2/pgt_summary.md runs/p2/pgt_q0_controls.json 2>/dev/null
