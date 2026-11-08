#!/usr/bin/env bash
# Serial V10 suite launcher.
#
# 1. waits for the five style caches
# 2. stage-1 gate: arm 04 trained from scratch, full 600 cross-style generation
# 3. remaining implemented arms on the Quick200 subset, strictly serial
#
# Progress is appended to runs/blog_sbflow/v10/suite_state.json after every arm.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
ROOT=runs/blog_sbflow/v10
STYLES="Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e"
mkdir -p "$ROOT/logs"

echo "[launch] waiting for five style caches ..."
while true; do
  n=0
  for S in $STYLES; do [ -f "runs/blog_sbflow/cache5/$S.pt" ] && n=$((n+1)); done
  [ "$n" -eq 5 ] && break
  echo "[launch] caches ready: $n/5 ($(date +%H:%M:%S))"
  sleep 120
done
echo "[launch] all five caches present"

echo "[launch] === stage-1 gate: arm 04, full600 ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 04 --eval-subset full600 --resume \
  2>&1 | tee -a "$ROOT/logs/gate.log"
GATE_RC=${PIPESTATUS[0]}
echo "[launch] gate exit=$GATE_RC"

if [ "$GATE_RC" -ne 0 ]; then
  echo "[launch] gate failed; stopping before mechanism arms"
  exit 1
fi

echo "[launch] === arms A (batch/sampling) ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 01-03 --eval-subset quick200 --resume \
  2>&1 | tee -a "$ROOT/logs/groupA.log"

echo "[launch] === arms B (parameterisation) ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 05-10 --eval-subset quick200 --resume \
  2>&1 | tee -a "$ROOT/logs/groupB.log"

echo "[launch] === arms C (bridge/EDM) ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 11,12,14,15 --eval-subset quick200 --resume \
  2>&1 | tee -a "$ROOT/logs/groupC.log"

echo "[launch] === arms G,H ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 37,38,39,40 --eval-subset quick200 --resume \
  2>&1 | tee -a "$ROOT/logs/groupGH.log"

echo "[launch] === arms D (distribution supervision) ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 16-23 --eval-subset quick200 --resume \
  2>&1 | tee -a "$ROOT/logs/groupD.log"

echo "[launch] === arms E (online alpha-DSBM) ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 24-31 --eval-subset quick200 --resume \
  2>&1 | tee -a "$ROOT/logs/groupE.log"

echo "[launch] === arms F (Gaussian pivot) ==="
python3 -u -m sbflow.run_suite --root "$ROOT" --arms 32-35 --eval-subset quick200 --resume \
  2>&1 | tee -a "$ROOT/logs/groupF.log"

echo "[launch] === ALL GROUPS DONE ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/blog_sbflow/v10/suite_state.json")
s = json.loads(p.read_text())
done = {k: v for k, v in s.items() if v.get("status") == "done"}
print(f"[launch] completed {len(done)}/{len(s)} arms")
for k in sorted(done):
    e = done[k].get("eval") or {}
    if e:
        print(f"  {k}: top1={e.get('target_top1', float('nan')):.4f} "
              f"margin={e.get('target_margin', float('nan')):+.4f} "
              f"dT={e.get('csd_dT', float('nan')):+.4f} swd={e.get('csd_swd', float('nan')):.4f}")
PY
