#!/bin/bash
# WEAVE-CPT / V7R_NEXT serial runner (single RTX 3060 12GB, WSL).
#
#   bash sbflow/scripts/run_v7r_next.sh --dry-run   # compile + CPU tests only
#   bash sbflow/scripts/run_v7r_next.sh            # full P0..P5 serial
#
# Design (per user habit): all components implemented & unit-tested first,
# ONE script drives the serial pipeline. Low coupling, let-it-crash, no overtest.
# Budget is RESEARCH budget (15h), NOT a GPU wall-clock ETA.
set -uo pipefail
R=/mnt/i/Latent_Style/WEAVE
cd "$R" || exit 1
PY=/usr/bin/python3
RUNS="$R/runs/v7r_next"
STATE="$RUNS/state"; LOGS="$RUNS/logs"
mkdir -p "$STATE" "$LOGS"
LOG="$LOGS/runner.log"
BUDGET_TOTAL_SEC=$((15*3600))
T0=$(date +%s)
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

log(){ echo "[v7r_next $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ log "FATAL $*"; exit 1; }
done_(){ [[ -f "$STATE/$1.done" ]]; }
mark(){ local s="$1"; shift; for a in "$@"; do [[ -e "$R/$a" ]] || fail "[$s] missing $a"; done;
        { date +%s; printf '%s\n' "$@"; } > "$STATE/$s.done"; log "[$s] marked done"; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %'); (( u < 25 )); }
budget_left(){ echo $(( BUDGET_TOTAL_SEC - ($(date +%s) - T0) )); }

run(){ local s="$1" n="$2"; shift 2; log "[$s] START $n: $*"; "$@" >> "$LOGS/${s}_${n}.log" 2>&1; local rc=$?;
       if (( rc != 0 )); then log "[$s] $n FAILED rc=$rc (tail)"; tail -n 25 "$LOGS/${s}_${n}.log" >&2;
         # NO early pipeline termination: record FAILED, continue with fallback.
         echo "$n" >> "$STATE/$s.failed"; return 0; fi
       log "[$s] DONE $n"; }

# ---------------------------------------------------------------- 0. preflight
log "V7R_NEXT start budget=${BUDGET_TOTAL_SEC}s dry=$DRY_RUN"
$PY -m py_compile tools/v6_next/r2_eval.py tools/v7r/arms.py \
    tools/v7r_next/*.py || fail "compile"
$PY -m tools.v7r_next.tests 2>&1 | tee "$LOGS/selftest.log" || fail "CPU selftests"
$PY - <<'PY' || fail "frozen input check"
import sys;sys.path.insert(0,"/mnt/i/Latent_Style/WEAVE")
from tools.v7r_next import common as C
need=[C.TASKS,C.REFS,C.B4_CKPT,C.R1_CKPT,C.PATCH_BANK]
for p in need:
    assert p.exists(), f"MISSING {p}"
print("frozen inputs OK", C.tasks_sha()[:16])
PY
if (( DRY_RUN )); then log "DRY-RUN complete (compile + 20 CPU tests + inputs). NOT starting GPU."; exit 0; fi

# ---------------------------------------------------------------- P0 (Gate)
stage_p0(){
  done_ p0 && { log "[p0] already done, skip"; return 0; }
  gpu_free || { log "[p0] GPU busy, skip (no early stop)"; return 0; }
  run p0 direction $PY -m tools.v7r_next.audit p0a_direction
  run p0 duplicate $PY -m tools.v7r_next.audit p0b_duplicate
  run p0 correction $PY -m tools.v7r_next.audit p0c_verdict
  # Gate
  $PY - <<'PY' || { log "[p0] GATE FAIL -> fallback: specificity mandatory"; echo "SPECIFICITY_MANDATORY" > "$STATE/p0.gate"; }
import sys,json;sys.path.insert(0,"/mnt/i/Latent_Style/WEAVE")
from tools.v7r_next import common as C
g=C.read_json(C.RUNS/"p0_gate.json")
print("p0 verdict",g["verdict"])
assert g["verdict"]=="PASS", "direction/branch error -> must fix & re-eval"
PY
  mark p0 runs/v7r_next/p0_gate.json runs/v7r_next/p0_audit/duplicate_audit.json
}
stage_p0

# ---------------------------------------------------------------- P1 (main line)
stage_p1(){
  done_ p1 && { log "[p1] already done, skip"; return 0; }
  gpu_free || return 0
  run p1 bank $PY -m tools.v7r_next.patch_transport build
  [[ -f "$RUNS/p1_bank.pt" ]] || { log "[p1] bank missing, fallback to E3 legacy bank"; echo "FALLBACK_E3" > "$STATE/p1.gate"; }
  # C1..C6 full-750 ablations + P1 Gate (geometry must beat uniform on specificity)
  run p1 evals $PY -m tools.v7r_next.run_evals p1
  mark p1 runs/v7r_next/p1_bank.pt runs/v7r_next/p1_bank_meta.json \
          runs/v7r_next/p1/p1_gate.json
}
stage_p1

# ---------------------------------------------------------------- P2 (gated)
stage_p2(){
  done_ p2 && { log "[p2] already done, skip"; return 0; }
  gpu_free || return 0
  run p2 rho $PY -m tools.v7r_next.corrector rho
  [[ -f "$RUNS/p2/rho_tiers.json" ]] || { log "[p2] rho unavailable -> P2 SKIPPED_BY_GATE"; echo "SKIPPED_BY_GATE" > "$STATE/p2.gate"; return 0; }
  # P1 Gate: geometry must be valuable. Now enforced by run_evals.p1_gate()
  # (C2/C3 specificity > C1). T2..T6 full 750 only if Gate passed.
  if $PY -c "import json,sys;g=json.load(open('runs/v7r_next/p1/p1_gate.json'));sys.exit(0 if g['geometry_valuable'] else 2)"; then
    run p2 evals $PY -m tools.v7r_next.run_evals p2
  else
    log "[p2] SKIPPED_BY_GATE: P1 geometry not demonstrated; keep R3c/R4c"
    echo "SKIPPED_BY_GATE" > "$STATE/p2.gate"
  fi
  mark p2 runs/v7r_next/p2/rho_tiers.json
}
stage_p2

# ---------------------------------------------------------------- P3 (gated)
stage_p3(){
  done_ p3 && { log "[p3] already done, skip"; return 0; }
  gpu_free || return 0
  run p3 teacher $PY -m tools.v7r_next.distill teacher --n 2000
  [[ -f "$RUNS/p3/teacher.pt" ]] || { log "[p3] teacher missing -> keep test-time line"; echo "TEACHER_FAILED" > "$STATE/p3.gate"; return 0; }
  run p3 D1 $PY -m tools.v7r_next.distill train --arm D1 --steps 3000
  run p3 D2 $PY -m tools.v7r_next.distill train --arm D2 --steps 3000
  run p3 evals $PY -m tools.v7r_next.run_evals p3
  if [[ -f "$RUNS/p3/teacher.pt" && -f "$RUNS/p3/D1/train_trace.json" ]]; then
    mark p3 runs/v7r_next/p3/teacher.pt runs/v7r_next/p3/D1/train_trace.json
  else
    log "[p3] incomplete -> keep TEST-TIME CPT line, no trained-corrector claim"
    echo "DISTILL_INCOMPLETE" > "$STATE/p3.gate"
  fi
}
stage_p3

# ---------------------------------------------------------------- P4 / P5 delegated to evaluate.py
stage_p45(){
  done_ p5 && { log "[p5] already done, skip"; return 0; }
  gpu_free || return 0
  # P5 builds eval_index from all completed arms, then report + speed + dual frontier.
  $PY - <<'PY' || log "[p5] index build failed (non-fatal)"
import sys,json;sys.path.insert(0,"/mnt/i/Latent_Style/WEAVE")
from pathlib import Path
from tools.v7r_next import common as C
idx={}
# history (read-only)
for name,rel in {
 "B3":"runs/v6_r2/eval_B3_distill20/eval.json",
 "B4":"runs/v6_r2/eval_B4_distill50/eval.json",
 "Q2a_R1_Oracle":"runs/v6_r2/eval_Q2a_oracle/eval.json",
 "R3c":"runs/v7r/p3_corrector/R3c/eval.json",
 "R4c":"runs/v7r/p3_corrector/R4c/eval.json",
 "R3c_disp":"runs/v7r/p3_corrector/R3c_disp/eval.json",
}.items():
    p=C.ROOT/rel
    if p.exists(): idx[name]=rel
# new CPT / student evals (added by their stages)
for p in sorted((C.RUNS).rglob("eval.json")):
    if "v7r_next" in str(p) and p.parent.name not in idx:
        idx[p.parent.name]=str(p.relative_to(C.ROOT))
C.write_json(C.RUNS/"p5/eval_index.json",idx)
print("eval_index arms:",list(idx))
PY
  run p5 report $PY -m tools.v7r_next.evaluate report
  run p5 speed  $PY -m tools.v7r_next.evaluate speed
  mark p5 runs/v7r_next/p5/final_report.json docs/v7/V7R_NEXT_BREAKTHROUGH.md
}
stage_p45

log "ALL V7R_NEXT STAGES FINISHED in $(( $(date +%s) - T0 ))s; budget_left $(budget_left)s"
