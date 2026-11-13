#!/bin/bash
# V8 queue round 3: P0-fix retrains + F-route (style actuation).
#
# Retrains arms whose results were invalidated by implementation bugs fixed in
# this round:
#   A3/A4  policy had no content input; A4 local branch was a deadlock
#   L3/L4  SPDHead/ControlSPDHead U=0 saddle -> low-rank branch untrainable
#   C3     critic trained on WRONG style labels (tid instead of sid)
# Eval arms depending on those checkpoints are deleted and re-run.
# L2 is NOT retrained: rank=0 means it has no U branch, fix does not apply.
#
# Then F0-F3 actuation training + eval (incl. F2 inference-step sweep).
set -u
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs/v8

LOG=logs/v8_queue_f.log
START_TS=$(date +%s)
LIMIT="${V8_LIMIT:-0}"
BATCH="${V8_BATCH:-8}"

log(){ echo "[v8f $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

arm_state(){ python3 - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path('runs/v8')/sys.argv[1]/'state.json'
print(json.loads(p.read_text()).get('state','') if p.exists() else '')
PY
}

run_arm(){
  local arm="$1"
  if [ "$(arm_state "$arm")" = "DONE" ]; then log "skip $arm (done)"; return 0; fi
  log "START $arm"
  local t0=$(date +%s)
  if python3 -m exp.v8.run_arm --arm "$arm" --limit "$LIMIT" --batch "$BATCH" \
        >> "logs/v8_${arm//\//_}.log" 2>&1; then
    log "DONE  $arm in $(( $(date +%s) - t0 ))s"
  else
    rc=$?
    log "FAILED $arm rc=$rc (queue continues)"
    python3 - "$arm" "$rc" <<'PY'
import json,sys
from pathlib import Path
p=Path('runs/v8')/sys.argv[1]/'state.json'
p.parent.mkdir(parents=True,exist_ok=True)
cur=json.loads(p.read_text()) if p.exists() else {}
cur.update(state='FAILED',returncode=int(sys.argv[2]))
p.write_text(json.dumps(cur,indent=2))
PY
  fi
}

train(){
  local a="$1"
  log "train $a"
  python3 - "$a" <<'PY' >> logs/v8_train_f.log 2>&1
import sys; sys.path.insert(0,'.')
from exp.v8.train import train_arm
train_arm(sys.argv[1])
PY
  log "train $a rc=$?"
}

# ------------------------------------------------ invalidate + retrain
# Old checkpoints/results for these arms were produced by buggy code; remove
# them so nothing silently reuses them.
for a in A3 A4 L3 L4 C3; do
  rm -rf "runs/v8/train/$a"
done
for a in A3.eval A4.eval I0 I1 I2 I3 C3o.r15 C3o.r25 C4 L3.e; do
  rm -rf "runs/v8/$a"
done
log "invalidated arms purged"

# ------------------------------------------------ retrain (fixed code)
for a in A3 A4 L3 L4 C3; do train "$a"; done

# ------------------------------------------------ F: style actuation
for a in F0 F1 F2 F3; do train "$a"; done

# ------------------------------------------------ inference
# invalidated eval arms
for a in A3.eval A4.eval I0 I1 I2 I3 C3o.r15 C3o.r25 C4 L3.e; do
  run_arm "$a"
done
# F eval
for a in F0.e F1.e F2.e F3.e F2.s1 F2.s4; do run_arm "$a"; done

log "QUEUE F END after $(( $(date +%s) - START_TS ))s"
