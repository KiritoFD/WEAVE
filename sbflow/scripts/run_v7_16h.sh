#!/usr/bin/env bash
# WEAVE V7 16h serial runner (docs/v7/V7_IMPLEMENTATION_PLAN.md section 0.3).
#
#   bash sbflow/scripts/run_v7_16h.sh             # full serial, resume-aware
#   bash sbflow/scripts/run_v7_16h.sh --dry-run   # compile + unit tests + input check
#
# - one stage at a time, single GPU exclusivity enforced before every stage;
# - stage completion markers: runs/v7/state/<stage>.done (with artifact list);
# - any failure stops the pipeline (no auto-skip to the next stage);
# - resume: stages already marked .done with intact artifacts are skipped;
# - budgets: V7_BUDGET_SEC (total, default 57600) and per-stage budgets.
set -uo pipefail

REPO="/mnt/i/Latent_Style/WEAVE"
RUNS="$REPO/runs/v7"
STATE="$RUNS/state"
LOGS="$RUNS/logs"
V7_BUDGET_SEC="${V7_BUDGET_SEC:-57600}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

cd "$REPO" || exit 1
mkdir -p "$STATE" "$LOGS"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-/usr/bin/python3}"
BATCH="${V7_BATCH:-8}"
T0_WALL=$(date +%s)

log()  { echo "[runner $(date +%H:%M:%S)] $*"; }
fail() { echo "[runner FATAL] $*" >&2; exit 1; }

gpu_free() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ -z "$apps" ]]
}

elapsed_ok() { # $1 = stage budget seconds, $2 = stage name
  local now start
  now=$(date +%s)
  start=$(cat "$STATE/.t_${2}" 2>/dev/null || echo "$T0_WALL")
  local used=$(( now - T0_WALL ))
  if (( used > V7_BUDGET_SEC )); then
    fail "total budget exceeded (${used}s > ${V7_BUDGET_SEC}s) at stage $1"
  fi
  local s_used=$(( now - start ))
  if (( s_used > $1 )); then
    fail "stage $2 exceeded its budget (${s_used}s > $1s)"
  fi
}

run_cmd() { # $1=stage $2=name $3=stage_budget_sec, rest = command
  local stage="$1" name="$2" budget="$3"; shift 3
  elapsed_ok "$budget" "$stage"
  log "[$stage] START $name: $*"
  "$@" >> "$LOGS/${stage}_${name}.log" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    tail -n 40 "$LOGS/${stage}_${name}.log" >&2 || true
    fail "[$stage] $name failed (rc=$rc); pipeline stopped"
  fi
  log "[$stage] DONE $name"
}

need_eval() { # skip if eval.json already exists (resume)
  [[ -f "$REPO/$1/eval.json" ]]
}

mark_done() { # $1=stage ; remaining args = artifacts to verify
  local stage="$1"; shift
  local a
  for a in "$@"; do
    [[ -e "$REPO/$a" ]] || fail "[$stage] completion artifact missing: $a"
  done
  { date +%s; printf '%s\n' "$@"; } > "$STATE/${stage}.done"
  log "[$stage] marked done"
}

stage_done() { [[ -f "$STATE/$1.done" ]] && { log "[$1] already done, skip"; return 0; }; return 1; }

# ---------------------------------------------------------------------------
preflight_inputs() {
  local missing=0 f p
  for f in \
    eval_protocol/tasks_750.json \
    eval_protocol/cache/references_v1_features.pt \
    runs/v6_r2/cache/tasks750_source_features.pt \
    runs/v6/P1_B4_distill50/P1_B4_distill50.pt \
    runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
    runs/v6_next/probes/r1_r1.pt \
    runs/v6/cache/canonicalizer_stats.pt \
    runs/v6_r2/eval_B3_distill20/eval.json \
    runs/v6_r2/eval_B4_distill50/eval.json \
    runs/v6_w/eval_topk_fix/fixed_topk25_b8/eval.json \
    /mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt \
    pseudo_gt_24h/orig_mean_fp16_a0/00_Early_Renaissance.pt ; do
    case "$f" in
      /*) p="$f" ;;
      *)  p="$REPO/$f" ;;
    esac
    if [[ ! -e "$p" ]]; then
      echo "MISSING $p"; missing=1
    fi
  done
  if ! gpu_free; then echo "GPU BUSY: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader)"; missing=1; fi
  return $missing
}

if (( DRY_RUN )); then
  log "dry-run: compiling tools/v7 + r2_eval"
  $PY -m py_compile \
    tools/v7/v7_common.py tools/v7/p0_causal_audit.py tools/v7/p1_energy.py \
    tools/v7/p1_validity.py tools/v7/p2_corrector.py tools/v7/p3_train.py \
    tools/v7/p4_combine.py tools/v7/p5_report.py tools/v7/p5_visual_grid.py \
    tools/v7/v7_tests.py tools/v6_next/r2_eval.py \
    || fail "compile failed"
  log "dry-run: unit tests (CPU)"
  CUDA_VISIBLE_DEVICES="" $PY -m tools.v7.v7_tests || fail "unit tests failed"
  log "dry-run: frozen input check"
  preflight_inputs || fail "input preflight failed"
  log "dry-run: OK (P0->P5 plan verified, no GPU task started)"
  exit 0
fi

preflight_inputs || fail "preflight failed (missing inputs or GPU busy)"

# ---------------------------------------------------------------------------
stage_p0() {
  stage_done p0 && return 0
  gpu_free || fail "GPU busy; single-GPU exclusivity violated"
  date +%s > "$STATE/.t_p0"
  local B=7200
  need_eval runs/v7/p0_relative/p0_q2a_direct || \
    run_cmd p0 q2a_direct $B $PY -m tools.v6_next.r2_eval --gen direct \
      --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
      --out-dir runs/v7/p0_relative/p0_q2a_direct --batch-size $BATCH
  need_eval runs/v7/p0_relative/p0_q2a_relative || \
    run_cmd p0 q2a_relative $B $PY -m tools.v6_next.r2_eval --gen relative \
      --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
      --out-dir runs/v7/p0_relative/p0_q2a_relative --batch-size $BATCH
  need_eval runs/v7/p0_relative/p0_b4_direct || \
    run_cmd p0 b4_direct $B $PY -m tools.v6_next.r2_eval --gen direct \
      --ckpt runs/v6/P1_B4_distill50/P1_B4_distill50.pt \
      --out-dir runs/v7/p0_relative/p0_b4_direct --batch-size $BATCH
  need_eval runs/v7/p0_relative/p0_b4_relative || \
    run_cmd p0 b4_relative $B $PY -m tools.v6_next.r2_eval --gen relative \
      --ckpt runs/v6/P1_B4_distill50/P1_B4_distill50.pt \
      --out-dir runs/v7/p0_relative/p0_b4_relative --batch-size $BATCH
  run_cmd p0 audit $B $PY -m tools.v7.p0_causal_audit
  mark_done p0 \
    runs/v7/p0_relative/p0_q2a_direct/eval.json \
    runs/v7/p0_relative/p0_q2a_relative/eval.json \
    runs/v7/p0_relative/p0_b4_direct/eval.json \
    runs/v7/p0_relative/p0_b4_relative/eval.json \
    runs/v7/p0_b4_causal_audit.json runs/v7/p0_style_delta.json \
    docs/v7/V7_P0_NOTE.md runs/v7/p0_verdict.json
}

stage_p1() {
  stage_done p1 && return 0
  gpu_free || fail "GPU busy"
  date +%s > "$STATE/.t_p1"
  local B=10800
  run_cmd p1 energy $B $PY -m tools.v7.p1_energy
  run_cmd p1 validity $B $PY -m tools.v7.p1_validity
  mark_done p1 runs/v7/p1_energy/energy_audit.json \
    runs/v7/p1_energy/energy_validity.json \
    runs/v7/p1_energy/S1_sd.pt runs/v7/p1_energy/S2_sd.pt
}

p1_go() {
  $PY - <<'EOF'
import json, sys
try:
    v = json.load(open("runs/v7/p1_energy/energy_validity.json"))
except Exception:
    sys.exit(1)
sys.exit(0 if v.get("verdict") == "GO_ENERGY" else 2)
EOF
}

stage_p2() {
  stage_done p2 && return 0
  gpu_free || fail "GPU busy"
  date +%s > "$STATE/.t_p2"
  local B=14400
  local CKPT="runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
  local v7args=(--gen v7 --ckpt "$CKPT" --batch-size $BATCH)
  need_eval runs/v7/p2_eval/T0 || \
    run_cmd p2 T0 $B $PY -m tools.v6_next.r2_eval "${v7args[@]}" --v7-arm T0 --out-dir runs/v7/p2_eval/T0
  need_eval runs/v7/p2_eval/T1 || \
    run_cmd p2 T1 $B $PY -m tools.v6_next.r2_eval "${v7args[@]}" --v7-arm T1 --out-dir runs/v7/p2_eval/T1
  if p1_go; then
    log "[p2] P1 GO_ENERGY -> running T2/T3/T4"
    for arm in T2 T3 T4; do
      need_eval "runs/v7/p2_eval/$arm" || \
        run_cmd p2 $arm $B $PY -m tools.v6_next.r2_eval "${v7args[@]}" --v7-arm $arm --out-dir "runs/v7/p2_eval/$arm"
    done
    # T5 only if T4 shows a positive signal vs T0 (predefined rule)
    if $PY - <<'EOF'
import json, sys
t0 = json.load(open("runs/v7/p2_eval/T0/eval.json"))["cross600"]
t4 = json.load(open("runs/v7/p2_eval/T4/eval.json"))["cross600"]
sys.exit(0 if (t4["dino_s"] > t0["dino_s"] - 0.002 and
               t4["dino_c"] >= t0["dino_c"] - 0.005) else 2)
EOF
    then
      need_eval runs/v7/p2_eval/T5 || \
        run_cmd p2 T5 $B $PY -m tools.v6_next.r2_eval "${v7args[@]}" --v7-arm T5 --out-dir runs/v7/p2_eval/T5
    else
      log "[p2] T4 has no positive signal; T5 skipped (predefined rule)"
      echo '{"skipped": ["T5"], "reason": "T4 no positive signal vs T0"}' > "$RUNS/p2_t5_skipped.json"
    fi
  else
    log "[p2] P1 NO_ENERGY -> only T0/T1 controls; writing verdict"
    cat > "$RUNS/p2_no_energy_verdict.json" <<'EOF'
{"verdict": "NO_ENERGY", "arms_run": ["T0", "T1"],
 "note": "P1 validity failed; T2-T5 not run per V7_EXPERIMENT_MATRIX.md section 3"}
EOF
  fi
  mark_done p2 runs/v7/p2_eval/T0/eval.json runs/v7/p2_eval/T0/grad_norm.json \
    runs/v7/p2_eval/T0/optim_trace.json runs/v7/p2_eval/T1/eval.json
}

stage_p3() {
  stage_done p3 && return 0
  gpu_free || fail "GPU busy"
  date +%s > "$STATE/.t_p3"
  local B=14400
  run_cmd p3 train_M1 $B $PY -m tools.v7.p3_train --name M1 --loss v6
  need_eval runs/v7/p3_eval/M1 || \
    run_cmd p3 eval_M1 $B $PY -m tools.v6_next.r2_eval --gen relative \
      --ckpt runs/v7/p3_models/M1/M1.pt --out-dir runs/v7/p3_eval/M1 --batch-size $BATCH
  if p1_go; then
    run_cmd p3 train_M2 $B $PY -m tools.v7.p3_train --name M2 --loss v7
    need_eval runs/v7/p3_eval/M2 || \
      run_cmd p3 eval_M2 $B $PY -m tools.v6_next.r2_eval --gen relative \
        --ckpt runs/v7/p3_models/M2/M2.pt --out-dir runs/v7/p3_eval/M2 --batch-size $BATCH
  else
    log "[p3] P1 NO_ENERGY -> M2 not trained (its loss is undefined)"
    echo '{"skipped": ["M2"], "reason": "P1 NO_ENERGY"}' > "$RUNS/p3_m2_skipped.json"
  fi
  local done_args=(runs/v7/p3_models/M1/M1.pt runs/v7/p3_models/M1/five_style_probe.json
    runs/v7/p3_models/M1/source_leak_probe.json runs/v7/p3_eval/M1/eval.json)
  if [[ -f "$RUNS/p3_models/M2/M2.pt" ]]; then
    done_args+=(runs/v7/p3_models/M2/M2.pt runs/v7/p3_models/M2/five_style_probe.json
      runs/v7/p3_eval/M2/eval.json)
  fi
  mark_done p3 "${done_args[@]}"
}

stage_p4() {
  stage_done p4 && return 0
  gpu_free || fail "GPU busy"
  date +%s > "$STATE/.t_p4"
  local B=5400
  run_cmd p4 plan $B $PY -m tools.v7.p4_combine plan
  local CKPT_B4="runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
  # C0/C1 are reuses (plan json lists them); run only C2/C3/C4 if planned
  local arm ckpt mode start
  for arm in C2 C3 C4; do
    need_eval "runs/v7/p4_eval/$arm" && continue
    if $PY - "$arm" <<'EOF'
import json, sys
plan = json.load(open("runs/v7/p4_plan.json"))["plan"]
sys.exit(0 if any(p["arm"] == sys.argv[1] and "gen" in p for p in plan) else 2)
EOF
    then
      ckpt=$CKPT_B4
      [[ $arm == "C3" || $arm == "C4" ]] && ckpt="runs/v7/p3_models/M2/M2.pt"
      mode=$($PY -c "import json;plan=json.load(open('runs/v7/p4_plan.json'))['plan'];print([p['gen'] for p in plan if p['arm']=='$arm'][0])")
      varm=$($PY -c "import json;plan=json.load(open('runs/v7/p4_plan.json'))['plan'];print([p.get('v7_arm','') for p in plan if p['arm']=='$arm'][0])")
      start=$($PY -c "import json;plan=json.load(open('runs/v7/p4_plan.json'))['plan'];print([p.get('start_mode','direct') for p in plan if p['arm']=='$arm'][0])")
      local extra=()
      [[ -n "$varm" ]] && extra+=(--v7-arm "$varm")
      run_cmd p4 $arm $B $PY -m tools.v6_next.r2_eval --gen "$mode" \
        "${extra[@]}" --v7-start-mode "$start" \
        --ckpt "$ckpt" --out-dir "runs/v7/p4_eval/$arm" --batch-size $BATCH
    fi
  done
  run_cmd p4 judge $B $PY -m tools.v7.p4_combine judge
  mark_done p4 runs/v7/p4_plan.json runs/v7/p4_verdict.json
}

stage_p5() {
  stage_done p5 && return 0
  gpu_free || fail "GPU busy"
  date +%s > "$STATE/.t_p5"
  local B=5400
  # visual grid: M2 (relative) if it exists, else B4 direct
  local grid_ckpt="runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
  local grid_mode="direct"
  if [[ -f "$RUNS/p3_models/M2/M2.pt" ]]; then
    grid_ckpt="runs/v7/p3_models/M2/M2.pt"; grid_mode="relative"
  fi
  run_cmd p5 grid $B $PY -m tools.v7.p5_visual_grid --ckpt "$grid_ckpt" --start-mode "$grid_mode"
  run_cmd p5 report $B $PY -m tools.v7.p5_report
  mark_done p5 runs/v7/final_matrix.json runs/v7/final_verdict.json \
    runs/v7/final_report_inputs.json runs/v7/runtime.json \
    runs/v7/visual_grid.png docs/v7/V7_REPORT.md
}

log "V7 serial runner start (budget ${V7_BUDGET_SEC}s)"
stage_p0
stage_p1
stage_p2
stage_p3
stage_p4
stage_p5
log "ALL STAGES COMPLETE in $(( $(date +%s) - T0_WALL ))s"
