#!/usr/bin/env bash
# =============================================================================
# WEAVE R2 — SINGLE SERIAL ENTRY POINT (12h budget)
# =============================================================================
# One script to launch the whole R2 plan, serially, on a single GPU.
#
#   bash sbflow/scripts/run_r2_12h.sh              # full run (resume-aware)
#   bash sbflow/scripts/run_r2_12h.sh --force      # ignore completion flags, recompute
#   bash sbflow/scripts/run_r2_12h.sh --dry-run    # preflight only, no GPU work
#
# Design (plan P5.2):
#   * Single-card exclusion: waits for GPU idle before every phase; never preempts.
#   * Atomic completion flags: runs/v6_r2/state/<PHASE>.done written only after
#     the phase's artefacts are verified. A killed run resumes from the last phase.
#   * Resume-aware: a phase whose flag + artefacts exist is SKIPPED (idempotent).
#   * Never overwrites historical checkpoints/results (out paths are R2-only).
#   * Records start/finish time, exit codes, model SHAs and per-phase wall clock.
#   * OOM downgrade: if a training phase OOMs, retry once with microbatch 40 / accum 6
#     (keeps effective batch 240).
#   * 12h hard budget: past the deadline, low-priority phases are skipped and the
#     stage report is still written from whatever completed.
#   * No concurrency: everything runs in this one foreground process, one phase at a time.
#
# All results go under runs/v6_r2/. The single formal report is docs/v6/R2_12H_REPORT.md.
# =============================================================================
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE

export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

R2=runs/v6_r2
STATE=$R2/state
LOG=$R2/queue_r2_12h.log
mkdir -p "$STATE" "$R2"

FORCE=0
DRY=0
for a in "$@"; do
  [ "$a" = "--force" ] && FORCE=1
  [ "$a" = "--dry-run" ] && DRY=1
done

# wall-clock budget (seconds). Default 12h from launch; overridable via R2_BUDGET_SEC.
BUDGET=${R2_BUDGET_SEC:-43200}
START_EPOCH=$(date +%s)
DEADLINE=$((START_EPOCH + BUDGET))

log () { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }

sha () { sha256sum "$1" 2>/dev/null | cut -d' ' -f1 | cut -c1-16 || echo "NA"; }

# phase_done <PHASE> : true if flag exists AND all listed artefacts are non-empty
phase_done () {
  local phase=$1; shift
  [ "$FORCE" -eq 1 ] && return 1
  [ -f "$STATE/$phase.done" ] || return 1
  for f in "$@"; do
    [ -s "$f" ] || { log "  resume: $phase flag present but artefact missing: $f"; return 1; }
  done
  return 0
}

mark_done () {
  local phase=$1; local t=$2; local rc=$3
  local tmp="$STATE/$phase.done.tmp.$$"
  {
    echo "phase=$phase"
    echo "finished=$(date +%Y-%m-%dT%H:%M:%S)"
    echo "elapsed_sec=$t"
    echo "exit_code=$rc"
  } > "$tmp" && mv -f "$tmp" "$STATE/$phase.done"   # atomic rename
}

# run_phase <PHASE> <expect1> [expect2...] -- <cmd...>
run_phase () {
  local phase=$1; shift
  local expects=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do expects+=("$1"); shift; done
  [ "$1" = "--" ] && shift

  if phase_done "$phase" "${expects[@]}"; then
    log "===== PHASE $phase : SKIP (already done, artefacts verified) ====="
    return 0
  fi

  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "===== PHASE $phase : SKIP (12h budget exhausted) ====="
    return 0
  fi

  log "===== PHASE $phase : START ====="
  local t0=$SECONDS
  wait_for_gpu 360 || { log "GPU busy, aborting phase $phase"; return 1; }

  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  local dt=$((SECONDS - t0))

  local missing=0
  for f in "${expects[@]}"; do
    if [ ! -s "$f" ]; then log "  PHASE $phase missing artefact: $f"; missing=1; fi
  done
  if [ "$rc" -ne 0 ] || [ "$missing" -ne 0 ]; then
    log "===== PHASE $phase : FAIL rc=$rc missing=$missing after ${dt}s (NO flag written) ====="
    FAILED_PHASES+=("$phase:$rc")
    return 1
  fi
  mark_done "$phase" "$dt" "$rc"
  log "===== PHASE $phase : OK in ${dt}s ====="
  return 0
}

source sbflow/scripts/q_lib.sh
FAILED_PHASES=()

# -----------------------------------------------------------------------------
# PREFLIGHT (P5.3): syntax + 12 unit tests. Cheap, CPU-only. Abort if anything fails.
# -----------------------------------------------------------------------------
log "################## WEAVE R2 SERIAL LAUNCH ##################"
log "budget=${BUDGET}s deadline=$(date -d @$DEADLINE +%H:%M:%S 2>/dev/null || echo $DEADLINE) force=$FORCE dry=$DRY"

log "----- PREFLIGHT: compile -----"
python3 -m py_compile \
  tools/v6_next/r2_eval.py tools/v6_next/r2_teacher.py \
  tools/v6_next/dino_s_control.py tools/v6_next/run_p1_analysis.py \
  tools/v6_next/run_p2_teachers.py tools/v6_next/student_fit_diag.py \
  tools/v6_next/paired_comparison_r2.py tools/v6_next/make_visual_grids.py \
  tools/v6_next/r2_unit_tests.py tools/v6/train_eval_v6.py || {
    log "PREFLIGHT COMPILE FAILED -- abort"; exit 2; }

log "----- PREFLIGHT: 12 unit tests -----"
if ! python3 -m tools.v6_next.r2_unit_tests 2>&1 | tee -a "$LOG"; then
  log "PREFLIGHT UNIT TESTS FAILED -- abort"; exit 2
fi

# Record the checkpoint SHAs used, for provenance (P3 engineering requirement)
{
  echo "launch=$(date +%Y-%m-%dT%H:%M:%S)"
  echo "tasks_sha256=$(sha eval_protocol/tasks_750.json)"
  echo "refs_sha256=$(sha eval_protocol/cache/references_v1_features.pt)"
  echo "q2a_ckpt_sha256=$(sha runs/v6/Q2a_embed_style/Q2a_embed_style.pt)"
  echo "critic_ckpt_sha256=$(sha runs/v6_next/probes/r1_r1.pt)"
  echo "score_diff_sha256=$(sha runs/v6/score_diff/score_diff_net.pt)"
  echo "seed=42"
} > "$R2/provenance.txt"

if [ "$DRY" -eq 1 ]; then
  log "################## DRY-RUN COMPLETE (no GPU work) ##################"
  exit 0
fi

# -----------------------------------------------------------------------------
# STATE BOOTSTRAP — recover completion flags for phases that were already run
# (P0-P4 were executed stage-by-stage before this single entry existed). A flag
# is written ONLY after its artefacts pass a real validity check (JSON parses,
# n_cross==600 for evals, checkpoint non-trivial size). This makes resume safe
# and avoids recomputing 12h of finished work.
# -----------------------------------------------------------------------------
bootstrap () {
  python3 - "$R2" "$STATE" <<'PY'
import json, sys
from pathlib import Path
R2, STATE = Path(sys.argv[1]), Path(sys.argv[2])
STATE.mkdir(parents=True, exist_ok=True)

EVALS = ["eval_identity", "eval_Q2a_direct", "eval_Q2a_oracle",
         "eval_B3_distill20", "eval_B4_distill50", "eval_scorediff_N24",
         "eval_S1", "eval_S2", "eval_S3", "eval_S4", "eval_P1_B2c_ctrl845"]

def eval_ok(name):
    p = R2 / name / "eval.json"
    if not p.exists() or p.stat().st_size < 1000: return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("n_cross") == 600 and len(d.get("rows", [])) == 750
    except Exception:
        return False

def json_ok(rel):
    p = R2 / rel
    if not p.exists() or p.stat().st_size < 200: return False
    try:
        json.loads(p.read_text(encoding="utf-8")); return True
    except Exception:
        return False

def ckpt_ok(rel):
    p = R2 / rel
    return p.exists() and p.stat().st_size > 500_000

def txt_has(rel, needle):
    p = R2 / rel
    return p.exists() and needle in p.read_text(encoding="utf-8", errors="ignore")

def flag(phase, ok, detail):
    f = STATE / f"{phase}.done"
    if f.exists(): print(f"  {phase}: flag exists"); return
    if ok:
        tmp = f.with_suffix(".done.tmp")
        tmp.write_text(f"phase={phase}\nfinished=bootstrapped\n"
                       f"elapsed_sec=0\nexit_code=0\nnote={detail}\n", encoding="utf-8")
        tmp.replace(f)
        print(f"  {phase}: BOOTSTRAPPED ({detail})")
    else:
        print(f"  {phase}: incomplete -> will RUN")

p0 = all(eval_ok(e) for e in EVALS[:6]) and txt_has("dino_s_control.txt", "5-Way Style Top-1 Acc")
flag("P0", p0, "6 baseline evals + DINO-S control verified")
flag("P1", json_ok("p1_analysis/p1_summary.json"), "P1 summary verified")
flag("P2", json_ok("p2_teachers/p2_teachers_summary.json"), "P2 summary verified")

p3_ckpts = ["S1_distill20_proj_ll/S1_distill20_proj_ll.pt",
            "S2_distill50_proj_ll/S2_distill50_proj_ll.pt",
            "S3_ctrl845_distill20_oracle/S3_ctrl845_distill20_oracle.pt",
            "S4_ctrl845_distill20_proj_ll/S4_ctrl845_distill20_proj_ll.pt"]
p3 = all(ckpt_ok(c) for c in p3_ckpts)
flag("P3", p3, "4 student checkpoints verified")

p4 = (all(eval_ok(e) for e in EVALS[6:])
      and txt_has("final_paired_report.txt", "PAIRED CLUSTERED BOOTSTRAP")
      and (R2 / "visual_grids/grid_balanced_5styles.png").exists())
flag("P4", p4, "4 student evals + 845K control + paired report + grids verified")
PY
}
if [ "$FORCE" -eq 1 ]; then
  export R2_FORCE=1
  log "----- STATE BOOTSTRAP: SKIP (--force, full serial rerun) -----"
else
  log "----- STATE BOOTSTRAP -----"
  bootstrap 2>&1 | tee -a "$LOG"
fi

# -----------------------------------------------------------------------------
# PHASE P0 — unified eval + 6 baselines + DINO-S validity control
# -----------------------------------------------------------------------------
P0_ARTS=(
  "$R2/eval_identity/eval.json"
  "$R2/eval_Q2a_direct/eval.json"
  "$R2/eval_Q2a_oracle/eval.json"
  "$R2/eval_B3_distill20/eval.json"
  "$R2/eval_B4_distill50/eval.json"
  "$R2/eval_scorediff_N24/eval.json"
  "$R2/dino_s_control.txt"
)
run_phase P0 "${P0_ARTS[@]}" -- bash -u sbflow/scripts/run_p0_diagnostics.sh

# -----------------------------------------------------------------------------
# PHASE P1 — R1 gradient mechanism & subband scan
# -----------------------------------------------------------------------------
run_phase P1 "$R2/p1_analysis/p1_summary.json" -- bash -u sbflow/scripts/run_p1_analysis.sh

# -----------------------------------------------------------------------------
# PHASE P2 — content-constrained teachers
# -----------------------------------------------------------------------------
run_phase P2 "$R2/p2_teachers/p2_teachers_summary.json" -- bash -u sbflow/scripts/run_p2_teachers.sh

# -----------------------------------------------------------------------------
# PHASE P3 — distillation matrix S1-S4 + fitting diagnostics (with OOM downgrade)
# -----------------------------------------------------------------------------
P3_ARTS=(
  "$R2/S1_distill20_proj_ll/S1_distill20_proj_ll.pt"
  "$R2/S2_distill50_proj_ll/S2_distill50_proj_ll.pt"
  "$R2/S3_ctrl845_distill20_oracle/S3_ctrl845_distill20_oracle.pt"
  "$R2/S4_ctrl845_distill20_proj_ll/S4_ctrl845_distill20_proj_ll.pt"
)
if ! phase_done P3 "${P3_ARTS[@]}"; then
  if [ "$(date +%s)" -lt "$DEADLINE" ]; then
    log "===== PHASE P3 : START ====="
    P3_T0=$SECONDS
    if ! bash -u sbflow/scripts/run_p3_distillation.sh 2>&1 | tee -a "$LOG"; then
      # OOM downgrade: one retry with microbatch 40 / accum 6 (effective batch still 240)
      log "P3 first attempt failed; checking for OOM to decide on downgrade retry"
      # Look only at the tail of this run's log so a historical OOM can't trigger a false retry.
      if tail -n 200 "$LOG" | grep -qi "out of memory"; then
        log "OOM detected -- retrying P3 with microbatch 40 / accum 6"
        sed 's/--microbatch 80 --accum 3/--microbatch 40 --accum 6/' \
          sbflow/scripts/run_p3_distillation.sh > "$R2/run_p3_downgraded.sh"
        bash -u "$R2/run_p3_downgraded.sh" 2>&1 | tee -a "$LOG"
      fi
    fi
    P3_OK=1
    for f in "${P3_ARTS[@]}"; do [ -s "$f" ] || P3_OK=0; done
    if [ "$P3_OK" -eq 1 ]; then
      bash -u sbflow/scripts/run_p3_fitting_diag.sh 2>&1 | tee -a "$LOG"
      mark_done P3 "$((SECONDS - P3_T0))" 0
      log "===== PHASE P3 : OK in $((SECONDS - P3_T0))s ====="
    else
      log "===== PHASE P3 : FAIL (missing checkpoints, no flag) ====="
      FAILED_PHASES+=("P3:missing_ckpt")
    fi
  else
    log "===== PHASE P3 : SKIP (budget) ====="
  fi
else
  log "===== PHASE P3 : SKIP (already done) ====="
fi

# -----------------------------------------------------------------------------
# PHASE P4 — final 750 eval of all students + paired bootstrap + visual grids
# -----------------------------------------------------------------------------
P4_ARTS=(
  "$R2/eval_S1/eval.json" "$R2/eval_S2/eval.json"
  "$R2/eval_S3/eval.json" "$R2/eval_S4/eval.json"
  "$R2/eval_P1_B2c_ctrl845/eval.json"
  "$R2/final_paired_report.txt"
  "$R2/visual_grids/grid_balanced_5styles.png"
)
run_phase P4 "${P4_ARTS[@]}" -- bash -u sbflow/scripts/run_p4_final_eval.sh

# -----------------------------------------------------------------------------
# PHASE P5 — single formal stage report (always written, even on partial results)
# -----------------------------------------------------------------------------
log "===== PHASE P5 : generating stage report ====="
python3 -u -m tools.v6_next.make_r2_report 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
TOTAL=$SECONDS   # bash monotonic counter; avoids NTP/WSL wall-clock skew
log "################## WEAVE R2 SERIAL RUN FINISHED ##################"
log "total_wall_sec=$TOTAL  failed_phases=${FAILED_PHASES[*]:-none}"
for d in "$STATE"/*.done; do
  [ -e "$d" ] && log "  done: $(basename "$d")  ($(grep elapsed_sec "$d" | cut -d= -f2)s)"
done
log "report: docs/v6/R2_12H_REPORT.md"
log "##############################################################"

[ "${#FAILED_PHASES[@]}" -eq 0 ] && exit 0 || exit 1
