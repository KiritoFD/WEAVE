#!/usr/bin/env bash
# Shared helpers for the autonomous P2 follow-up queue.
#
# The previous queue used `set -uo pipefail` with `... | tee`, so a training failure was masked:
# the pipeline status is tee's, and nothing aborted. Every step here goes through run_step, which
#   * captures the REAL exit code via PIPESTATUS[0] before tee can overwrite it,
#   * refuses to continue when the step fails,
#   * verifies the artefacts the step was supposed to produce, so "TRAINING DONE" alone can never
#     be read as success.
set -uo pipefail

QLOG=${QLOG:-runs/v6_24h/queue_q.log}
QFAIL=0

qlog () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

# run_step <label> <expect-file-or-empty> <cmd...>
run_step () {
  local label=$1; local expect=$2; shift 2
  qlog "===== STEP $label ====="
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$QLOG"
  local rc=${PIPESTATUS[0]}
  local dt=$((SECONDS - t0))
  if [ "$rc" -ne 0 ]; then
    qlog "STEP $label FAILED rc=$rc after ${dt}s"
    QFAIL=$((QFAIL+1))
    return 1
  fi
  if [ -n "$expect" ]; then
    if [ ! -s "$expect" ]; then
      qlog "STEP $label produced no $expect -- treating as FAILURE despite rc=0"
      QFAIL=$((QFAIL+1))
      return 1
    fi
  fi
  qlog "STEP $label OK rc=0 in ${dt}s${expect:+ (verified $expect)}"
  return 0
}

# verify_arm <name>: a training arm is only "done" if the checkpoint AND report are valid JSON
verify_arm () {
  local name=$1
  local ck="runs/v6/${name}/${name}.pt"
  local rp="runs/v6/${name}/report.json"
  if [ ! -s "$ck" ]; then qlog "  ARM $name MISSING checkpoint"; return 1; fi
  if [ ! -s "$rp" ]; then qlog "  ARM $name MISSING report"; return 1; fi
  python3 - "$rp" <<'PY' || { qlog "  ARM report invalid"; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
e = d.get("eval200") or {}
assert e.get("mean_dT") is not None, "no mean_dT"
assert d.get("n_params"), "no n_params"
PY
  qlog "  ARM $name OK ($(du -h "$ck" | cut -f1))"
  return 0
}

wait_for_gpu () {
  local max_polls=${1:-360}
  local i=0
  while [ "$i" -lt "$max_polls" ]; do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n1)" ]; then
      qlog "GPU free after $i polls"
      return 0
    fi
    i=$((i+1))
    sleep 20
  done
  qlog "GPU still busy after $max_polls polls -- aborting rather than preempting"
  return 1
}
