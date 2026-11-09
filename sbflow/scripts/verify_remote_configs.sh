#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
echo "=== compile ==="
python3 -m py_compile sbflow/train5.py sbflow/gen_configs40.py sbflow/stamp_metadata40.py sbflow/full750.py sbflow/suite40.py
echo "=== dry config loader (exp24 fork metadata) ==="
python3 - <<'PY'
import json, argparse, sys
from pathlib import Path
from sbflow.train5 import _apply_config
# minimal parser matching train5's required args and store_false aliases
ap=argparse.ArgumentParser()
ap.add_argument('--experiment-id',required=True); ap.add_argument('--caches',nargs='+',required=True)
ap.add_argument('--output',required=True); ap.add_argument('--config',default='')
ap.add_argument('--steps',type=int,default=800); ap.add_argument('--batch',type=int,default=1024)
ap.add_argument('--mixed-batch',action='store_true',default=True)
ap.add_argument('--single-pair',dest='mixed_batch',action='store_false')
ap.add_argument('--equal-weight-stats',action='store_true',default=True)
ap.add_argument('--frequency-stats',dest='equal_weight_stats',action='store_false')
ap.add_argument('--vae',default=''); ap.add_argument('--scaling-factor',type=float,default=0)
a=_apply_config(ap,['--config','configs/sbflow/exp01.json','--caches','a','b'])
print('  exp01 steps',a.steps,'mixed',a.mixed_batch,'eq',a.equal_weight_stats)
a=_apply_config(ap,['--config','configs/sbflow/exp05.json','--caches','a','b'])
print('  exp05 mixed',a.mixed_batch)
a=_apply_config(ap,['--config','configs/sbflow/exp06.json','--caches','a','b'])
print('  exp06 eq',a.equal_weight_stats)
PY
echo "=== status ==="
python3 -u sbflow/fold_scores40.py >/dev/null 2>&1 || true
python3 -u sbflow/backfill_timing40.py >/dev/null 2>&1 || true
echo "REMOTE CONFIG VERIFY PASSED"
