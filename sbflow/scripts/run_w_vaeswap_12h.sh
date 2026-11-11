#!/usr/bin/env bash
# =============================================================================
# WEAVE VAE-SWAP — single serial entry (A0 re-encode + retrain + eval + compare)
# =============================================================================
# Answers: "重新编码，换 VAE 训练看看". Kept as a SEPARATE serial script so it
# never disturbs the already-running W pipeline; launch it AFTER W finishes
# (single-GPU exclusion via wait_for_gpu).
#
#   bash sbflow/scripts/run_w_vaeswap_12h.sh           # full serial (resume-aware)
#   bash sbflow/scripts/run_w_vaeswap_12h.sh --dry-run # preflight only
# =============================================================================
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

W=runs/v6_w
STATE=$W/state_vaeswap
LOG=$W/queue_w_vaeswap_12h.log
mkdir -p "$STATE" "$W"
DRY=0; FORCE=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; [ "$a" = "--force" ] && FORCE=1; done
BUDGET=${V_BUDGET_SEC:-43200}
START=$(date +%s); DEADLINE=$((START+BUDGET))

log () { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }
phase_done () {
  local phase=$1; shift
  [ "$FORCE" -eq 1 ] && return 1
  [ -f "$STATE/$phase.done" ] || return 1
  for f in "$@"; do [ -s "$f" ] || return 1; done
  return 0
}
mark_done () { local p=$1; { echo "phase=$p"; echo "finished=$(date +%Y-%m-%dT%H:%M:%S)"; } > "$STATE/$p.done.tmp.$$" && mv -f "$STATE/$p.done.tmp.$$" "$STATE/$p.done"; }
run_phase () {
  local phase=$1; shift
  local ex=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do ex+=("$1"); shift; done
  [ "$1" = "--" ] && shift
  if phase_done "$phase" "${ex[@]}"; then log "===== $phase SKIP (done) ====="; return 0; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then log "===== $phase SKIP (budget) ====="; return 0; fi
  log "===== $phase START ====="
  wait_for_gpu 360 || return 1
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  for f in "${ex[@]}"; do [ -s "$f" ] || { log "  missing $f"; rc=1; }; done
  if [ "$rc" -ne 0 ]; then log "===== $phase FAIL rc=$rc ====="; FAILED+=("$phase:$rc"); return 1; fi
  mark_done "$phase"; log "===== $phase OK ====="
}
source sbflow/scripts/q_lib.sh
FAILED=()

log "################## VAE-SWAP SERIAL LAUNCH ##################"
log "----- PREFLIGHT -----"
python3 -m py_compile \
  tools/v6_next/reencode_a0.py tools/v6_next/w_vae.py \
  tools/v6/train_eval_v6.py tools/v6_next/r2_eval.py \
  tools/v6_next/w_paired_compare.py tools/v6_next/w_report.py || {
    log "COMPILE FAILED"; exit 2; }
python3 -m tools.v6_next.w_unit_tests 2>&1 | tee -a "$LOG" || { log "UNIT TESTS FAILED"; exit 2; }
python3 - <<'PY' || { echo "A0 ckpt missing" | tee -a "$LOG"; exit 2; }
from tools.v6_next.w_vae import _resolve, A0_ENCODER_CANDIDATES
p = _resolve(A0_ENCODER_CANDIDATES, "A0")
print(f"  A0 encoder: {p} ({p.stat().st_size:,} bytes)")
PY
if [ "$DRY" -eq 1 ]; then log "################## DRY-RUN COMPLETE ##################"; exit 0; fi

V0=(pseudo_gt_24h/orig_mean_fp16_a0/reencode_manifest.json
    runs/v6_w/V1_A0_LossClean/V1_A0_LossClean.pt
    runs/v6_w/V2_A0_PixelMSE/V2_A0_PixelMSE.pt)
run_phase V0 "${V0[@]}" -- bash -u sbflow/scripts/run_w_vaeswap_train.sh

V1=(runs/v6_w/eval_V1_A0_LossClean/eval.json
    runs/v6_w/eval_V2_A0_PixelMSE/eval.json)
run_phase V1 "${V1[@]}" -- bash -u sbflow/scripts/run_w_vaeswap_eval.sh

# V2: comparison + report (always)
log "===== V2 compare + report ====="
python3 -u -m tools.v6_next.w_paired_compare > "$W/w_paired_report.txt" 2>&1
python3 -u -m tools.v6_next.w_report 2>&1 | tee -a "$LOG"

log "################## VAE-SWAP RUN FINISHED ##################"
log "failed_phases=${FAILED[*]:-none}"
log "report: docs/v6/W_REPORT.md"
[ "${#FAILED[@]}" -eq 0 ] && exit 0 || exit 1
