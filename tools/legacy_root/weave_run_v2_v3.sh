#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
LOG=runs/v2/run_v2_v3_750.log
mkdir -p runs/v2/eval runs/v2/logs

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

run_train() {
  local variant=$1
  log "=== TRAIN $variant start ==="
  python3 -m v2.train --config "configs/v2/${variant}.json" --variant "$variant" \
    --steps 1600 --batch-size 144 --accumulation-steps 1 --save-at 500,1000,1600 --diag-every 250 \
    >> "runs/v2/logs/train_${variant}.log" 2>&1
  local rc=$?
  log "=== TRAIN $variant exit=$rc ==="
}

# 750 all-pairs protocol: B/32, decode-batch 8, 8 steps, B/32 CLIP.
run_eval() {
  local ck=$1 variant=$2 step=$3
  if [ ! -f "$ck" ]; then log "SKIP eval (missing $ck)"; return; fi
  log "=== EVAL $variant step $step start ==="
  python3 tools/eval_fast.py --checkpoint "$ck" \
    --output "runs/v2/eval/${variant}_step_${step}_750" \
    --device cuda --batch 32 --metric-batch 32 --decode-batch 8 --num-steps 8 \
    >> "runs/v2/logs/eval_${variant}_${step}.log" 2>&1
  local rc=$?
  log "=== EVAL $variant step $step exit=$rc ==="
}

run_train V2
run_eval runs/v2/V2/step_000500.pt V2 000500
run_eval runs/v2/V2/step_001000.pt V2 001000
run_eval runs/v2/V2/step_001600.pt V2 001600

run_train V3
run_eval runs/v2/V3/step_000500.pt V3 000500
run_eval runs/v2/V3/step_001000.pt V3 001000
run_eval runs/v2/V3/step_001600.pt V3 001600

log "=== ALL DONE ==="
