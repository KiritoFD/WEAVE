#!/bin/bash
# STAGE1 runner: fixed deterministic queue (no teacher cache, no calibration).
# Launched once; each job is an independent process so one failure never stops
# the rest. Progress and status live in atomic state files.
set -u
ROOT=/mnt/i/Latent_Style/WEAVE
REV="$ROOT/runs/v8_24h/deploy/run_r2"
export WEAVE_ROOT="$ROOT" PYTHONPATH="$REV:$ROOT"
cd "$REV" || exit 1
mkdir -p "$ROOT/runs/v8_24h/logs"
LOG="$ROOT/runs/v8_24h/logs/stage1.log"
exec >> "$LOG" 2>&1
printf 'STAGE1 START '; date -Is
# Fixed rho/tau tiers: user decision, recorded explicitly as not-measured.
python3 - <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'/mnt/i/Latent_Style/WEAVE/runs/v8_24h/deploy/run_r2')
from tools.v8_24h.core.common import atomic_json
p=Path('/mnt/i/Latent_Style/WEAVE/runs/v8_24h/preflight/calibration.json')
cal=json.loads(p.read_text()) if p.exists() else {}
cal.update({'rho_lo':0.08,'rho_hi':0.16,'tau_lo':1.0,'tau_hi':4.0,'epsilon':0.1,
            'content_slack':0.001,'rho_status':'FIXED_TIER_NOT_MEASURED',
            'note':'user decision: fixed tiers instead of a measured distribution'})
atomic_json(p,cal);print('fixed tiers written')
PY
python3 -m tools.v8_24h.core.launch --freeze --stage stage1 || exit $?
python3 -m tools.v8_24h.core.launch
rc=$?
printf 'STAGE1 END rc=%s ' "$rc"; date -Is
exit "$rc"
