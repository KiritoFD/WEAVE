#!/usr/bin/env bash
# =============================================================================
# WEAVE W P1 — SINGLE SERIAL ENTRY (real-latent score prior + guided editing)
# =============================================================================
#   bash sbflow/scripts/run_w_p1_12h.sh              # full serial (resume-aware)
#   bash sbflow/scripts/run_w_p1_12h.sh --dry-run    # preflight only
#
# Phases:
#   P1A train unconditional real-latent score prior on A0 latents
#   P1B evaluate guided editing: R1 only / score only / R1+score
#   P1C paired clustered bootstrap comparison and P1 report
#
# Uses atomic .done flags in runs/v6_w/state_p1 and a 12h budget gate.
# =============================================================================
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

W=runs/v6_w
STATE=$W/state_p1
LOG=$W/queue_w_p1_12h.log
mkdir -p "$STATE" "$W"
DRY=0; FORCE=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; [ "$a" = "--force" ] && FORCE=1; done
BUDGET=${P1_BUDGET_SEC:-43200}
START=$(date +%s); DEADLINE=$((START+BUDGET))

log () { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }
phase_done () {
  local phase=$1; shift
  [ "$FORCE" -eq 1 ] && return 1
  [ -f "$STATE/$phase.done" ] || return 1
  for f in "$@"; do [ -s "$f" ] || return 1; done
  return 0
}
mark_done () {
  local phase=$1 dt=$2
  { echo "phase=$phase"; echo "finished=$(date +%Y-%m-%dT%H:%M:%S)";
    echo "elapsed_sec=$dt"; } > "$STATE/$phase.done.tmp.$$" \
    && mv -f "$STATE/$phase.done.tmp.$$" "$STATE/$phase.done"
}
run_phase () {
  local phase=$1; shift
  local ex=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do ex+=("$1"); shift; done
  [ "$1" = "--" ] && shift
  if phase_done "$phase" "${ex[@]}"; then log "===== $phase SKIP (done) ====="; return 0; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then log "===== $phase SKIP (budget) ====="; return 0; fi
  log "===== $phase START ====="
  local t0=$SECONDS
  wait_for_gpu 360 || return 1
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]} dt=$((SECONDS-t0))
  local missing=0
  for f in "${ex[@]}"; do [ -s "$f" ] || { log "missing $f"; missing=1; }; done
  if [ "$rc" -ne 0 ] || [ "$missing" -ne 0 ]; then
    log "===== $phase FAIL rc=$rc missing=$missing ====="; FAILED+=("$phase:$rc"); return 1
  fi
  mark_done "$phase" "$dt"; log "===== $phase OK in ${dt}s ====="
}
source sbflow/scripts/q_lib.sh
FAILED=()

log "################## WEAVE W P1 SERIAL LAUNCH ##################"
log "----- PREFLIGHT: compile -----"
python3 -m py_compile \
  tools/v6_next/w_score_prior.py tools/v6_next/w_guided.py \
  tools/v6_next/r2_eval.py tools/v6_next/w_paired_compare.py \
  tools/v6_next/w_unit_tests.py || { log "COMPILE FAILED"; exit 2; }
log "----- PREFLIGHT: 14 unit tests -----"
python3 -m tools.v6_next.w_unit_tests 2>&1 | tee -a "$LOG" || { log "UNIT TESTS FAILED"; exit 2; }
log "----- PREFLIGHT: required inputs -----"
python3 - <<'PY' || { log "INPUT CHECK FAILED"; exit 2; }
from pathlib import Path
from tools.v6_next.w_vae import _resolve, A0_ENCODER_CANDIDATES, S10_DECODER_CANDIDATES
a = _resolve(A0_ENCODER_CANDIDATES, "A0 encoder")
s = _resolve(S10_DECODER_CANDIDATES, "S10 decoder")
pool = Path("pseudo_gt_24h/orig_mean_fp16_a0/reencode_manifest.json")
print(f"A0 encoder: {a} ({a.stat().st_size:,} bytes)")
print(f"S10 decoder: {s} ({s.stat().st_size:,} bytes)")
print(f"A0 latent pool manifest: {pool} ({pool.stat().st_size:,} bytes)")
PY

if [ "$DRY" -eq 1 ]; then log "################## DRY-RUN COMPLETE ##################"; exit 0; fi

run_phase P1A "$W/score_prior/score_prior.pt" "$W/score_prior/train_report.json" -- \
  bash -u sbflow/scripts/run_w_score_prior_train.sh

run_phase P1B \
  "$W/eval_guided_r1/eval.json" \
  "$W/eval_guided_score/eval.json" \
  "$W/eval_guided_both/eval.json" -- \
  bash -u sbflow/scripts/run_w_guided_eval.sh

run_phase P1C "$W/w_p1_compare_report.txt" "$W/w_p1_verdict.json" -- \
  bash -c 'python3 -u -m tools.v6_next.w_paired_compare > runs/v6_w/w_p1_compare_report.txt 2>&1 && python3 -u -m tools.v6_next.w_p1_report'

TOTAL=$SECONDS
log "################## WEAVE W P1 SERIAL RUN FINISHED ##################"
log "total_wall_sec=$TOTAL failed_phases=${FAILED[*]:-none}"
log "report: docs/v6/W_P1_REPORT.md"
[ "${#FAILED[@]}" -eq 0 ] && exit 0 || exit 1
