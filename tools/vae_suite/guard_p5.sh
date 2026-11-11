#!/bin/bash
# Guard: wait for the in-flight run_sd0_s10.sh to finish, then AUTOMATICALLY
# chain run_p5_final.sh (P5 is part of the same 48h plan, just split into a
# separate file so the running chain's bytes were never modified).
#
# Usage (launch once, it exits by itself):
#   nohup bash tools/vae_suite/guard_p5.sh > runs/sd0_s10/guard.log 2>&1 &
#
# Single-GPU safe: this guard itself uses NO GPU; it only polls a PID. P5 starts
# strictly AFTER the training chain has exited, so no two training/eval jobs
# ever compete for the 24 GB card.
set -uo pipefail
R=/root/Workspace/WEAVE_VAE
cd "$R"
OUT="$R/runs/sd0_s10"
LOCK="$OUT/.p5_guard.lock"
P=/opt/conda/envs/cu121/bin/python

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[guard] another guard already running; exit"; exit 0
fi

log() { echo "[guard $(date '+%F %T')] $*"; }

# Exclude this script itself (its own cmdline contains the pattern) by matching
# the exact anchored command of the controller.
CHAIN_PAT='^bash tools/vae_suite/run_sd0_s10.sh'
log "started; watching chain '$CHAIN_PAT'"

while pgrep -f "$CHAIN_PAT" > /dev/null 2>&1; do
  sleep 120
done
log "chain process gone"

sleep 20   # let the last tee/last.pt flush settle

CHAIN_OK=0
if grep -q "SERIAL PIPELINE COMPLETE" "$OUT/serial.log" 2>/dev/null; then
  log "chain completed cleanly"; CHAIN_OK=1
elif [ -f "$OUT/FINAL_REPORT.json" ]; then
  log "chain produced FINAL_REPORT.json (treat as completed)"; CHAIN_OK=1
else
  # Partial failure (e.g. C1 OOM). The plan requires: "评测脚本失败不丢权重"
  # and "Gate 失败保存负面报告". c0_best.pt is copied by the controller right
  # after C0 finishes (BEFORE C1 starts), so it survives a later C1 crash.
  # run_p5_final.sh builds MODELS from whichever best_*.pt files EXIST, so a
  # conservative C0-only P5 is still deliverable. Do NOT throw away C0.
  log "chain did NOT complete (no COMPLETE marker, no FINAL_REPORT.json)"
  tail -n 40 "$OUT/serial.log"
  # The controller copies c0_W_best.pt -> c0_best.pt right after C0 finishes.
  # If it died mid-C0, only c0_W_best.pt exists; recover it so P5 can still
  # deliver a conservative C0-only result (weights are never lost).
  if [ ! -f "$OUT/c0_best.pt" ] && [ -f "$OUT/c0_W_best.pt" ]; then
    cp "$OUT/c0_W_best.pt" "$OUT/c0_best.pt"
    log "recovered c0_W_best.pt -> c0_best.pt (chain died mid/late C0)"
  fi
  if [ -f "$OUT/c0_best.pt" ]; then
    log "c0_best.pt EXISTS -> writing NEGATIVE_REPORT and running conservative P5"
    {
      echo "{\"status\": \"CHAIN_INCOMPLETE_C1_OR_LATER\","
      echo " \"note\": \"training chain exited without COMPLETE; C0 weights preserved; P5 evaluates only existing checkpoints\","
      echo " \"c0_best_exists\": true,"
      echo " \"c1_best_exists\": $([ -f "$OUT/c1_best.pt" ] && echo true || echo false),"
      echo " \"timestamp\": \"$(date -Is)\"}"
    } > "$OUT/NEGATIVE_REPORT.json"
    CHAIN_OK=2   # degraded-but-deliverable
  else
    log "c0_best.pt MISSING too -> nothing deliverable; NOT starting P5"
    exit 3
  fi
fi
echo "$CHAIN_OK" > "$OUT/.chain_status"

if [ -f "$OUT/.p5_done" ]; then
  log "P5 already done"; exit 0
fi

log "launching P5 (run_p5_final.sh)"
bash tools/vae_suite/run_p5_final.sh > "$OUT/p5_guard_run.log" 2>&1
P5=$?
if [ "$P5" -eq 0 ]; then
  : > "$OUT/.p5_done"
  log "P5 finished OK (marker .p5_done written)"
else
  log "P5 exited $P5 (weights preserved; inspect p5_guard_run.log)"
fi
log "guard done"
exit "$P5"
