#!/bin/bash
# WEAVE V8 24h serial queue.
#
# Order (from the plan): P0 contract -> A train -> A infer -> L train -> C train
# -> C infer -> B infer -> I joint. Dependencies are explicit; a scientific NO
# never cancels a later arm. A crashing arm is recorded FAILED and the queue
# continues, because the remaining arms are independent.
#
# Resume: every arm writes runs/v8/<arm>/state.json; DONE arms are skipped.
set -u
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs/v8

LOG=logs/v8_queue.log
STATE=runs/v8/queue_state.json
START_TS=$(date +%s)
LIMIT="${V8_LIMIT:-0}"          # 0 = full 750
BATCH="${V8_BATCH:-8}"

log(){ echo "[v8 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

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
  python3 - "$STATE" "$arm" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {"done":[]}
d.setdefault("done",[]).append({"arm":sys.argv[2],"at":time.strftime("%H:%M:%S")})
p.write_text(json.dumps(d,indent=2))
PY
}

# ---------------------------------------------------------------- P0 contract
log "P0 contract"
python3 -m exp.v8.p0_contract >> logs/v8_p0.log 2>&1 \
  && log "P0 PASS" || log "P0 FAILED (recorded; queue continues)"

# ---------------------------------------------------------------- support bank
# The C route and the A/I arms that use lambda_ratio != 0 need the real-painting
# support prototypes in frozen R1 embedding space. Built from the TRAIN pool.
if [ -f runs/v8/real_support.pt ]; then
  log "support bank already present"
else
  log "building real support bank"
  python3 -m exp.v8.build_support >> logs/v8_support.log 2>&1 \
    && log "support bank OK" || log "support bank FAILED (C arms will be recorded FAILED)"
fi

# ---------------------------------------------------------------- training
# A: control policy (feeds A3.eval/A4.eval and the I arms)
for a in A3 A4; do
  python3 - <<PY >> logs/v8_train.log 2>&1
import sys; sys.path.insert(0,'.')
from exp.v8.train import train_arm
train_arm("$a")
PY
  log "train $a rc=$?"
done
# B: SPD operators (L4 needs A3's control space)
for a in L2 L3 L4; do
  python3 - <<PY >> logs/v8_train.log 2>&1
import sys; sys.path.insert(0,'.')
from exp.v8.train import train_arm
train_arm("$a")
PY
  log "train $a rc=$?"
done
# C: critic robustness
python3 - <<'PY' >> logs/v8_train.log 2>&1
import sys; sys.path.insert(0,'.')
from exp.v8.train import train_arm
train_arm("C3")
PY
log "train C3 rc=$?"

# ---------------------------------------------------------------- inference
# A inference (no training needed) then A3/A4 driven arms
for a in A0 A0.s2 A0.s4 A1 A1.s2 A1.s4 A2 A2.s2 A2.s4 A2b A3.eval A4.eval; do
  run_arm "$a"
done
# C inference
for a in C0 C1.l010 C1.l030 C2 C4 C3o.r15 C3o.r25; do run_arm "$a"; done
# B inference
for a in L0.r08 L0.r15 L0.r25 L1.b4.r25 L1.b4.r15 L1.topk L2.e L3.e; do run_arm "$a"; done
# I joint
for a in I0 I1 I2 I3; do run_arm "$a"; done

python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
from exp.v8.arms import GROUPS
runs=Path('runs/v8')
done=failed=missing=0
for g,arms in GROUPS.items():
    for a in arms:
        p=runs/a['id']/'state.json'
        st=json.loads(p.read_text()).get('state','') if p.exists() else ''
        if st=='DONE': done+=1
        elif st=='FAILED': failed+=1
        else: missing+=1
print(json.dumps({'done':done,'failed':failed,'missing':missing},indent=2))
PY
log "QUEUE END after $(( $(date +%s) - START_TS ))s"
