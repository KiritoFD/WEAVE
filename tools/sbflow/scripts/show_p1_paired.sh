#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import json
import numpy as np
from pathlib import Path
from tools.v6_24h.stats import paired_cluster_ci

BASE = "P1_B0_ctrl218"
ARMS = ["P1_B1_distill218", "P1_B2c_ctrl845", "P1_B2_distill845",
        "P1_B3_distill20", "P1_B4_distill50"]

def rows(a):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        return None
    return (json.loads(p.read_text()).get("eval200") or {}).get("rows")

r0 = rows(BASE)
c0 = [r["src_stem"] for r in r0]
print("PAIRED SOURCE-CLUSTERED DIFFERENCES vs", BASE)
print(f"  {'arm':20s}{'d(dT)':>10s}{'CI':>24s}{'excl0':>7s}{'d(Top1)':>10s}{'CI':>24s}{'excl0':>7s}")
for a in ARMS:
    ra = rows(a)
    if ra is None:
        print(f"  {a:20s}  missing"); continue
    dt = paired_cluster_ci([r["dT"] for r in ra], [r["dT"] for r in r0], c0)
    t1 = paired_cluster_ci([float(r["target_top1"]) for r in ra],
                           [float(r["target_top1"]) for r in r0], c0)
    print(f"  {a:20s}{dt['diff']:+10.4f}"
          f"{'[%+.4f,%+.4f]' % (dt['ci'][0], dt['ci'][1]):>24s}"
          f"{str(dt['excludes_zero']):>7s}"
          f"{t1['diff']:+10.4f}"
          f"{'[%+.4f,%+.4f]' % (t1['ci'][0], t1['ci'][1]):>24s}"
          f"{str(t1['excludes_zero']):>7s}")
PY
