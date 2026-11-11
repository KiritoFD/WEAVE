#!/usr/bin/env bash
# =============================================================================
# WEAVE W — SINGLE SERIAL ENTRY POINT
# =============================================================================
# Frozen Encoder / latent space. Attacks the R2 finding that students did NOT
# fit the Oracle teacher (cos 0.64, 55% rel error, 1.3+ margin gap).
#
#   bash sbflow/scripts/run_w_12h.sh              # full serial (resume-aware)
#   bash sbflow/scripts/run_w_12h.sh --dry-run    # preflight only, no GPU work
#   bash sbflow/scripts/run_w_12h.sh --force      # ignore completion flags
#
# Phases (all serial, single GPU):
#   W0 train matrix (4 arms) -> W1 750 eval -> W2 paired compare -> W3 select best
#   -> W4 S10 image-domain calibration -> W5 eval calibrated -> W6 report
#
# Atomic .done flags in runs/v6_w/state make it resume-safe; 12h budget gate;
# single-card exclusion; never overwrites R2 artefacts.
# =============================================================================
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE

export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

W=runs/v6_w
STATE=$W/state
LOG=$W/queue_w_12h.log
mkdir -p "$STATE" "$W"

FORCE=0; DRY=0
for a in "$@"; do
  [ "$a" = "--force" ] && FORCE=1
  [ "$a" = "--dry-run" ] && DRY=1
done

BUDGET=${W_BUDGET_SEC:-43200}
START=$(date +%s)
DEADLINE=$((START + BUDGET))

log () { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }

phase_done () {
  local phase=$1; shift
  [ "$FORCE" -eq 1 ] && return 1
  [ -f "$STATE/$phase.done" ] || return 1
  for f in "$@"; do
    [ -s "$f" ] || { log "  resume: $phase missing $f"; return 1; }
  done
  return 0
}

mark_done () {
  local phase=$1 dt=$2 rc=$3
  { echo "phase=$phase"; echo "finished=$(date +%Y-%m-%dT%H:%M:%S)";
    echo "elapsed_sec=$dt"; echo "exit_code=$rc"; } > "$STATE/$phase.done.tmp.$$" \
    && mv -f "$STATE/$phase.done.tmp.$$" "$STATE/$phase.done"
}

run_phase () {
  local phase=$1; shift
  local expects=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do expects+=("$1"); shift; done
  [ "$1" = "--" ] && shift
  if phase_done "$phase" "${expects[@]}"; then
    log "===== PHASE $phase : SKIP (done) ====="; return 0; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "===== PHASE $phase : SKIP (budget) ====="; return 0; fi
  log "===== PHASE $phase : START ====="
  local t0=$SECONDS
  wait_for_gpu 360 || { log "GPU busy, abort $phase"; return 1; }
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]} dt=$((SECONDS - t0))
  local missing=0
  for f in "${expects[@]}"; do [ -s "$f" ] || { log "  missing $f"; missing=1; }; done
  if [ "$rc" -ne 0 ] || [ "$missing" -ne 0 ]; then
    log "===== PHASE $phase : FAIL rc=$rc missing=$missing (no flag) ====="
    FAILED+=("$phase:$rc"); return 1
  fi
  mark_done "$phase" "$dt" "$rc"
  log "===== PHASE $phase : OK in ${dt}s ====="
}

source sbflow/scripts/q_lib.sh
FAILED=()

log "################## WEAVE W SERIAL LAUNCH ##################"
log "budget=${BUDGET}s force=$FORCE dry=$DRY"

# ---------------- PREFLIGHT ----------------
log "----- PREFLIGHT: compile -----"
python3 -m py_compile \
  tools/v6_next/w_distill.py tools/v6_next/w_vae.py tools/v6_next/w_img_calib.py \
  tools/v6_next/w_unit_tests.py tools/v6_next/w_select_best.py \
  tools/v6_next/w_paired_compare.py tools/v6_next/w_report.py \
  tools/v6_next/r2_eval.py tools/v6/train_eval_v6.py || {
    log "PREFLIGHT COMPILE FAILED -- abort"; exit 2; }

log "----- PREFLIGHT: 14 unit tests -----"
python3 -m tools.v6_next.w_unit_tests 2>&1 | tee -a "$LOG" || {
  log "PREFLIGHT UNIT TESTS FAILED -- abort"; exit 2; }

log "----- PREFLIGHT: A0/S10 checkpoint presence -----"
python3 - <<'PY' || { echo "A0/S10 check failed" | tee -a "$LOG"; exit 2; }
from tools.v6_next.w_vae import _resolve, A0_ENCODER_CANDIDATES, S10_DECODER_CANDIDATES
a = _resolve(A0_ENCODER_CANDIDATES, "A0")
s = _resolve(S10_DECODER_CANDIDATES, "S10")
print(f"  A0 encoder : {a} ({a.stat().st_size:,} bytes)")
print(f"  S10 decoder: {s} ({s.stat().st_size:,} bytes)")
PY

if [ "$DRY" -eq 1 ]; then
  log "################## DRY-RUN COMPLETE ##################"; exit 0
fi

# ---------------- W0: training matrix ----------------
W0_ARTS=(
  "$W/W1_FeatDistill_218/W1_FeatDistill_218.pt"
  "$W/W2_LossClean_218/W2_LossClean_218.pt"
  "$W/W3_LossClean_218_l50/W3_LossClean_218_l50.pt"
)
run_phase W0 "${W0_ARTS[@]}" -- bash -u sbflow/scripts/run_w_train.sh

# ---------------- W1: 750 eval of 4 arms ----------------
W1_ARTS=(
  "$W/eval_W1_FeatDistill_218/eval.json"
  "$W/eval_W2_LossClean_218/eval.json"
  "$W/eval_W3_LossClean_218_l50/eval.json"
)
run_phase W1 "${W1_ARTS[@]}" -- bash -u sbflow/scripts/run_w_eval.sh

# ---------------- W2: paired compare + W3: select best ----------------
run_phase W23 "$W/w_paired_verdict.json" "$W/w_best_arm.json" -- \
  bash -u sbflow/scripts/run_w_compare.sh

# ---------------- W4: S10 image-domain calibration ----------------
run_phase W4 "$W/W5_ImgCalib/W5_ImgCalib.pt" -- bash -u sbflow/scripts/run_w_imgcal.sh

# ---------------- W5: eval the calibrated model ----------------
run_phase W5 "$W/eval_W5_ImgCalib/eval.json" -- \
  python3 -u -m tools.v6_next.r2_eval \
    --gen direct --ckpt "$W/W5_ImgCalib/W5_ImgCalib.pt" \
    --out-dir "$W/eval_W5_ImgCalib"

# ---------------- W6: final report (always) ----------------
log "===== PHASE W6 : report ====="
python3 -u -m tools.v6_next.w_paired_compare > "$W/w_paired_report.txt" 2>&1
python3 -u -m tools.v6_next.w_report 2>&1 | tee -a "$LOG"

TOTAL=$SECONDS
log "################## WEAVE W SERIAL RUN FINISHED ##################"
log "total_wall_sec=$TOTAL failed_phases=${FAILED[*]:-none}"
for d in "$STATE"/*.done; do [ -e "$d" ] && log "  done: $(basename "$d")"; done
log "report: docs/v6/W_REPORT.md"
[ "${#FAILED[@]}" -eq 0 ] && exit 0 || exit 1
