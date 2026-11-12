#!/usr/bin/env bash
# WEAVE V7-R / V7-R+ 14h serial runner (docs/v7/V7R_PLAN.md + V7-R+ addendum).
#
#   bash sbflow/scripts/run_v7r_14h.sh             # full serial, resume-aware
#   bash sbflow/scripts/run_v7r_14h.sh --dry-run   # compile + CPU unit tests + inputs
#
# Execution principles:
#   - single-GPU exclusivity before every stage;
#   - atomic .done markers, checkpoint resume, BUDGET_TOTAL_SEC=57600 hard cap;
#   - a failed hypothesis switches to the pre-registered alternative branch;
#   - arms marked NO / SKIPPED_BY_GATE / SKIPPED_BY_BUDGET / MISSING;
#   - ETA accounting: BUDGET_TOTAL / ELAPSED / GPU_ACTIVE / ETA_REMAINING;
#   - stage order (V7-R+ section 7):
#       P0/P1 -> P2(N1/N2) -> E1 -> E2 -> E3 -> E5 -> E4 -> P3 -> P4 -> P5
#   - priority if the budget runs out:
#       P0 > N1/N2 > E2 > E3 > E1 > E5 > E4 > final report;
#     the report and full evaluations of existing candidates are never skipped.
set -uo pipefail

REPO="/mnt/i/Latent_Style/WEAVE"
RUNS="$REPO/runs/v7r"
STATE="$RUNS/state"
LOGS="$RUNS/logs"
BUDGET_TOTAL_SEC="${V7R_BUDGET_SEC:-57600}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

cd "$REPO" || exit 1
mkdir -p "$STATE" "$LOGS"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-/usr/bin/python3}"
BATCH="${V7R_BATCH:-8}"
T0_WALL=$(date +%s)

log()  { echo "[v7r $(date +%H:%M:%S)] $*"; }
fail() { echo "[v7r FATAL] $*" >&2; exit 1; }
skip_gate() { # $1 = arm dir, $2 = marker name, $3 = reason
  mkdir -p "$REPO/$1"
  printf '{"status": "%s", "reason": "%s"}\n' "$2" "$3" > "$REPO/$1/$2.json"
  log "[gate] $1 $2: $3"
}

gpu_free() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ -z "$apps" ]]
}

eta_update() {
  "$PY" -m tools.v7r.eta >> "$LOGS/eta.log" 2>&1 || true
  [[ -f "$RUNS/eta.json" ]] && \
    log "ETA: $($PY -c "import json;d=json.load(open('$RUNS/eta.json'));print('elapsed',d['ELAPSED_SEC'],'gpu_active~',d['GPU_ACTIVE_SEC_EST'],'remaining_work',d['ETA_REMAINING_SEC'],'budget_left',d['BUDGET_REMAINING_SEC'])" 2>/dev/null || echo n/a)"
}

budget_left() { echo $(( BUDGET_TOTAL_SEC - ($(date +%s) - T0_WALL) )) ; }

guard_budget() { # $1 = stage name, $2 = estimated GPU seconds
  local left
  left=$(( BUDGET_TOTAL_SEC - ($(date +%s) - T0_WALL) ))
  if (( left < $2 + 3600 )); then
    skip_gate "$RUNS/$1" "SKIPPED_BY_BUDGET" \
      "remaining ${left}s < estimate $2s + 1h report reserve (priority rule)"
    return 1
  fi
  return 0
}

run_cmd() { # $1=stage $2=name, rest = command
  local stage="$1" name="$2"; shift 2
  log "[$stage] START $name: $*"
  "$@" >> "$LOGS/${stage}_${name}.log" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    tail -n 40 "$LOGS/${stage}_${name}.log" >&2 || true
    fail "[$stage] $name failed (rc=$rc); pipeline stopped"
  fi
  log "[$stage] DONE $name"
}

eval_arm() { # $1=arm_dir $2=v7r_arm $3=ckpt $4=stage_name
  local dir="$1" arm="$2" ckpt="$3" stage="$4"
  if [[ -f "$REPO/$dir/eval.json" ]]; then log "[$stage] $dir already done, skip"; return 0; fi
  if [[ -f "$REPO/$dir/SKIPPED_BY_GATE.json" || -f "$REPO/$dir/SKIPPED_BY_BUDGET.json" ]]; then
    log "[$stage] $dir gate/budget-skipped"; return 0; fi
  run_cmd "$stage" "$(basename "$dir")" $PY -m tools.v6_next.r2_eval --gen v7r \
    --v7r-arm "$arm" --ckpt "$ckpt" --out-dir "$dir" --batch-size $BATCH
}

mark_done() { # $1=stage ; remaining = artifacts
  local stage="$1"; shift
  local a
  for a in "$@"; do
    [[ -e "$REPO/$a" || -e "$REPO/$a/SKIPPED_BY_GATE.json" || -e "$REPO/$a/SKIPPED_BY_BUDGET.json" ]] \
      || fail "[$stage] artifact missing: $a"
  done
  { date +%s; printf '%s\n' "$@"; } > "$STATE/${stage}.done"
  log "[$stage] marked done"
  eta_update
}

stage_done() { [[ -f "$STATE/$1.done" ]] && { log "[$1] already done, skip"; return 0; }; return 1; }

preflight_inputs() {
  local missing=0 f p
  for f in \
    eval_protocol/tasks_750.json \
    eval_protocol/cache/references_v1_features.pt \
    runs/v6_r2/cache/tasks750_source_features.pt \
    runs/v6/P1_B4_distill50/P1_B4_distill50.pt \
    runs/v6_next/probes/r1_r1.pt \
    runs/v6/cache/canonicalizer_stats.pt \
    runs/v6/cache/indomain_style_targets_v2_k64.pt \
    runs/v6_r2/eval_B3_distill20/eval.json \
    runs/v6_r2/eval_B4_distill50/eval.json \
    runs/v7/p1_energy/S2_sd.pt \
    runs/v7/p1_energy/S1_sd.pt \
    runs/v7/p0_relative/p0_b4_direct/eval.json \
    /mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt \
    /mnt/i/Latent_Style/pseudo_gt_24h/dino_224/Early_Renaissance.pt ; do
    case "$f" in /*) p="$f" ;; *) p="$REPO/$f" ;; esac
    [[ -e "$p" ]] || { echo "MISSING $p"; missing=1; }
  done
  if ! gpu_free; then echo "GPU BUSY: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader)"; missing=1; fi
  return $missing
}

if (( DRY_RUN )); then
  log "dry-run: compiling tools/v7r + p3_train + r2_eval"
  $PY -m py_compile \
    tools/v7r/__init__.py tools/v7r/v7r_common.py tools/v7r/arms.py \
    tools/v7r/p0_audit.py tools/v7r/p1_patchbank.py tools/v7r/p2_train.py \
    tools/v7r/e3_repr.py tools/v7r/gates.py tools/v7r/p4_decision.py \
    tools/v7r/p5_report.py tools/v7r/p5_grid.py tools/v7r/eta.py \
    tools/v7r/v7r_tests.py tools/v7/p3_train.py tools/v6_next/r2_eval.py \
    || fail "compile failed"
  log "dry-run: CPU unit tests"
  CUDA_VISIBLE_DEVICES="" $PY -m tools.v7r.v7r_tests || fail "unit tests failed"
  log "dry-run: frozen input check"
  preflight_inputs || fail "input preflight failed"
  log "dry-run: OK (V7-R+ plan verified, no GPU task started)"
  exit 0
fi

preflight_inputs || fail "preflight failed (missing inputs or GPU busy)"
"$PY" -m tools.v7r.eta --start >> "$LOGS/eta.log" 2>&1 || true
B4CKPT="runs/v6/P1_B4_distill50/P1_B4_distill50.pt"

# ---------------------------------------------------------------------------
stage_p0() {
  stage_done p0 && return 0
  gpu_free || fail "GPU busy"
  if [[ ! -d "$RUNS/p0_ckpts/M1" || -z "$(ls -A "$RUNS/p0_ckpts/M1" 2>/dev/null)" ]]; then
    run_cmd p0 retrain_M1 $PY -m tools.v7.p3_train --name M1 --loss v6 \
      --ckpt-steps 0,100,400,800 --out-dir runs/v7r/p0_ckpts/M1
  fi
  if [[ ! -d "$RUNS/p0_ckpts/M2" || -z "$(ls -A "$RUNS/p0_ckpts/M2" 2>/dev/null)" ]]; then
    run_cmd p0 retrain_M2 $PY -m tools.v7.p3_train --name M2 --loss v7 \
      --ckpt-steps 0,100,400,800 --out-dir runs/v7r/p0_ckpts/M2
  fi
  run_cmd p0 audit $PY -m tools.v7r.p0_audit
  mark_done p0 runs/v7r/p0_failure_audit.json docs/v7/V7R_FAILURE_AUDIT.md
}

p1_local_go() {
  $PY - <<'EOF'
import json, sys
try:
    v = json.load(open("runs/v7r/p1_energy/p1_gate.json"))
except Exception:
    sys.exit(1)
sys.exit(0 if v.get("verdict") == "GO_LOCAL" else 2)
EOF
}

stage_p1() {
  stage_done p1 && return 0
  gpu_free || fail "GPU busy"
  run_cmd p1 patchbank $PY -m tools.v7r.p1_patchbank
  for o in S1 S2 R1; do for v in fwd rev rand wrong; do
    eval_arm "runs/v7r/p3_corrector/p1a_${o}_${v}" "p1a_${o}_${v}" "$B4CKPT" p1
  done; done
  for e in E0 E1 E2 E3 E4; do
    eval_arm "runs/v7r/p3_corrector/p1c_${e}" "$e" "$B4CKPT" p1
  done
  run_cmd p1 gate $PY -m tools.v7r.gates p1
  if ! p1_local_go; then
    skip_gate "runs/v7r/p3_corrector/R3c" "SKIPPED_BY_GATE" "P1 gate NO_LOCAL"
    skip_gate "runs/v7r/p3_corrector/R4c" "SKIPPED_BY_GATE" "P1 gate NO_LOCAL"
    skip_gate "runs/v7r/p3_corrector/R3c_noC" "SKIPPED_BY_GATE" "P1 gate NO_LOCAL"
    skip_gate "runs/v7r/p3_corrector/R3c_disp" "SKIPPED_BY_GATE" "P1 gate NO_LOCAL"
  fi
  mark_done p1 runs/v7r/p1_energy/patch_banks.pt \
    runs/v7r/p1_energy/patch_bank_audit.json runs/v7r/p1_energy/p1_gate.json
}

p2_leakage_confirmed() {
  $PY - <<'EOF'
import json, sys
try:
    a = json.load(open("runs/v7r/p0_failure_audit.json"))
    frac = a["b4_shortcut"]["overall"].get("csd_pred_is_source_frac", 0)
except Exception:
    frac = 0.71
sys.exit(0 if frac > 0.5 else 2)
EOF
}

eval_all_ckpts() { # $1 = arm dir name (relative to p2_predictor)
  local n="$1"
  for ck in "$RUNS"/p2_predictor/$n/step_*.pt; do
    [[ -e "$ck" ]] || continue
    s=$(basename "$ck" .pt)
    eval_arm "runs/v7r/p2_predictor/$n/eval_${s}" "NPred" \
      "runs/v7r/p2_predictor/$n/${s}.pt" "p2eval_$n"
  done
}

stage_p2() {
  stage_done p2 && return 0
  gpu_free || fail "GPU busy"
  local SOBJ="v6dist"
  if p1_local_go; then
    SOBJ=$($PY -c "import json;print(json.load(open('runs/v7r/p1_energy/p1_gate.json'))['best_energy'])")
    log "[p2] using P1-gated style objective: $SOBJ"
  else
    log "[p2] P1 NO_LOCAL -> v6dist supervision (predefined fallback)"
  fi
  run_cmd p2 train_N1 $PY -m tools.v7r.p2_train --name N1 --mode corrector \
    --style-objective "$SOBJ"
  run_cmd p2 train_N2 $PY -m tools.v7r.p2_train --name N2 --mode corrector \
    --style-objective "$SOBJ"
  eval_all_ckpts N1
  eval_all_ckpts N2
  run_cmd p2 gate $PY -m tools.v7r.gates p2
  if ! $PY - <<'EOF'
import json, sys
try:
    v = json.load(open("runs/v7r/p2_predictor/p2_gate.json"))
except Exception:
    sys.exit(2)
sys.exit(2 if v.get("verdict") == "NEW_PREDICTOR" else 0)
EOF
  then
    log "[p2] N1/N2 failed -> checking alternative route gate"
    if p2_leakage_confirmed; then
      run_cmd p2 train_RAND $PY -m tools.v7r.p2_train --name RAND --mode rand \
        --ckpt-steps 0,1500,3000
      for ck in "$RUNS"/p2_predictor/RAND/step_*.pt; do
        [[ -e "$ck" ]] || continue
        s=$(basename "$ck" .pt)
        if [[ ! -f "$REPO/runs/v7r/p2_predictor/RAND/eval_${s}/eval.json" ]]; then
          run_cmd p2 eval_RAND_$s $PY -m tools.v6_next.r2_eval --gen direct \
            --ckpt "runs/v7r/p2_predictor/RAND/${s}.pt" \
            --out-dir "runs/v7r/p2_predictor/RAND/eval_${s}" --batch-size $BATCH
        fi
      done
    else
      skip_gate "runs/v7r/p2_predictor/RAND" "SKIPPED_BY_GATE" "leakage gate not confirmed"
    fi
  fi
  mark_done p2 runs/v7r/p2_predictor/N1/step_03000.pt \
    runs/v7r/p2_predictor/N2/step_03000.pt runs/v7r/p2_predictor/p2_gate.json
}

best_corrector_ckpt() {
  $PY -c "import json,sys;v=json.load(open('runs/v7r/p2_predictor/p2_gate.json'));print(v.get('best_ckpt') or '')" 2>/dev/null
}

stage_e1() {
  stage_done e1 && return 0
  gpu_free || fail "GPU busy"
  guard_budget e1 2700 || { mark_done e1 "runs/v7r/e1"; return 0; }
  local CK
  CK=$(best_corrector_ckpt)
  if [[ -z "$CK" || ! -f "$REPO/$CK" ]]; then
    skip_gate "runs/v7r/p3_corrector/E1_A" "SKIPPED_BY_GATE" "requires a trained corrector (p2_gate)"
    skip_gate "runs/v7r/p3_corrector/E1_B" "SKIPPED_BY_GATE" "requires a trained corrector (p2_gate)"
    skip_gate "runs/v7r/p3_corrector/E1_C" "SKIPPED_BY_GATE" "requires a trained corrector (p2_gate)"
    skip_gate "runs/v7r/p3_corrector/E1_D" "SKIPPED_BY_GATE" "requires a trained corrector (p2_gate)"
    skip_gate "runs/v7r/p3_corrector/E1_R1src" "SKIPPED_BY_GATE" "requires a trained corrector (p2_gate)"
    run_cmd e1 gate $PY -m tools.v7r.gates e1
    mark_done e1 "runs/v7r/e1_summary.json"
    return 0
  fi
  for arm in E1_A E1_B E1_C E1_D E1_R1src; do
    eval_arm "runs/v7r/p3_corrector/$arm" "$arm" "$CK" e1
  done
  run_cmd e1 gate $PY -m tools.v7r.gates e1
  mark_done e1 runs/v7r/e1_summary.json
}

stage_e2() {
  stage_done e2 && return 0
  gpu_free || fail "GPU busy"
  guard_budget e2 7200 || { mark_done e2 "runs/v7r/e2"; return 0; }
  # E2-A = the original N2 recipe (control reused, no new training)
  mkdir -p "$RUNS/p2_predictor/E2_A"
  printf '{"status": "REUSED", "note": "E2-A == original N2 training (same recipe/inputs); see p2_predictor/N2"}\n' \
    > "$RUNS/p2_predictor/E2_A/REUSED.json"
  run_cmd e2 train_E2B $PY -m tools.v7r.p2_train --name E2_B --mode corrector \
    --augment rand --consistency 0 --ckpt-steps 0,1000,2000,3000
  run_cmd e2 train_E2C $PY -m tools.v7r.p2_train --name E2_C --mode corrector \
    --augment rand --consistency 1 --ckpt-steps 0,1000,2000,3000
  eval_all_ckpts E2_B
  eval_all_ckpts E2_C
  run_cmd e2 gate $PY -m tools.v7r.gates p2
  mark_done e2 runs/v7r/p2_predictor/E2_B/step_03000.pt \
    runs/v7r/p2_predictor/E2_C/step_03000.pt \
    runs/v7r/p2_predictor/E2_B/augment_precheck.json
}

stage_e3() {
  stage_done e3 && return 0
  gpu_free || fail "GPU busy"
  guard_budget e3 9000 || { mark_done e3 "runs/v7r/e3"; return 0; }
  run_cmd e3 train_A $PY -m tools.v7r.e3_repr --variant A
  run_cmd e3 train_B $PY -m tools.v7r.e3_repr --variant B
  eval_arm "runs/v7r/p3_corrector/p1c_E3A_repr" "E3A_repr" "$B4CKPT" e3
  eval_arm "runs/v7r/p3_corrector/p1c_E3B_repr" "E3B_repr" "$B4CKPT" e3
  run_cmd e3 regate $PY -m tools.v7r.gates p1
  mark_done e3 runs/v7r/e3/repr_A.pt runs/v7r/e3/repr_B.pt \
    runs/v7r/e3/phase1_A.json runs/v7r/e3/phase1_B.json
}

stage_e5() {
  stage_done e5 && return 0
  gpu_free || fail "GPU busy"
  guard_budget e5 4500 || { mark_done e5 "runs/v7r/e5"; return 0; }
  run_cmd e5 train_E5 $PY -m tools.v7r.p2_train --name E5 --mode corrector \
    --core-frac 0.0 --ckpt-steps 0,1000,2000,3000
  eval_all_ckpts E5
  run_cmd e5 gate $PY -m tools.v7r.gates p2
  mark_done e5 runs/v7r/p2_predictor/E5/step_03000.pt
}

stage_e4() {
  stage_done e4 && return 0
  gpu_free || fail "GPU busy"
  guard_budget e4 7200 || { mark_done e4 "runs/v7r/e4"; return 0; }
  run_cmd e4 decide $PY -m tools.v7r.p4_decision decide
  local MODE
  MODE=$($PY -c "import json;print(json.load(open('$RUNS/p4_joint/p4_decision.json'))['mode'])")
  case "$MODE" in
    synergy)
      log "[e4] new predictor exists -> no E4 branch needed (synergy handled in P4)"
      ;;
    EXTEND)
      local n
      n=$($PY -c "import json;print(json.load(open('$RUNS/p4_joint/p4_decision.json'))['target_arm'])")
      run_cmd e4 extend_train $PY -m tools.v7r.p2_train --name "$n" \
        --mode corrector --steps 6000 --ckpt-steps 4000,5000,6000 \
        --init-corrector "runs/v7r/p2_predictor/$n/step_03000.pt"
      for s in 04000 05000 06000; do
        eval_arm "runs/v7r/p2_predictor/$n/eval_step_$s" "NPred" \
          "runs/v7r/p2_predictor/$n/step_$s.pt" e4
      done
      run_cmd e4 gate $PY -m tools.v7r.gates p2
      ;;
    ROLLBACK_LR)
      local n
      n=$($PY -c "import json;print(json.load(open('$RUNS/p4_joint/p4_decision.json'))['target_arm'])")
      local best
      best=$($PY -c "
import json, glob
best, bv = None, -1
for p in sorted(glob.glob('runs/v7r/p2_predictor/$n/eval_step_*/eval.json')):
    d = json.load(open(p))['cross600']['dino_s']
    if d > bv: best, bv = p, d
print(best.split('/')[-2].replace('eval_',''))")
      run_cmd e4 rollback_train $PY -m tools.v7r.p2_train --name "${n}_lr" \
        --mode corrector --steps 1500 --lr 1e-4 \
        --ckpt-steps 500,1000,1500 \
        --init-corrector "runs/v7r/p2_predictor/$n/${best}.pt"
      for ck in "$RUNS"/p2_predictor/${n}_lr/step_*.pt; do
        [[ -e "$ck" ]] || continue
        s=$(basename "$ck" .pt)
        eval_arm "runs/v7r/p2_predictor/${n}_lr/eval_${s}" "NPred" \
          "runs/v7r/p2_predictor/${n}_lr/${s}.pt" e4
      done
      run_cmd e4 gate $PY -m tools.v7r.gates p2
      ;;
    ALTERNATIVE)
      local n
      n=$($PY -c "import json;print(json.load(open('$RUNS/p4_joint/p4_decision.json'))['target_arm'])")
      if [[ "$n" == "E2_C" ]]; then
        run_cmd e4 alt_train $PY -m tools.v7r.p2_train --name E2_C --mode corrector \
          --augment rand --consistency 1 --ckpt-steps 0,1000,2000,3000
        eval_all_ckpts E2_C
      elif [[ "$n" == "E5" ]]; then
        run_cmd e4 alt_train $PY -m tools.v7r.p2_train --name E5 --mode corrector \
          --core-frac 0.0 --ckpt-steps 0,1000,2000,3000
        eval_all_ckpts E5
      else
        run_cmd e4 alt_train $PY -m tools.v7r.p2_train --name E2_B --mode corrector \
          --augment rand --ckpt-steps 0,1000,2000,3000
        eval_all_ckpts E2_B
      fi
      run_cmd e4 gate $PY -m tools.v7r.gates p2
      ;;
    *)
      log "[e4] no branch selected ($MODE)"
      ;;
  esac
  # second seed (E4 rule): only when a positive improvement exists
  if $PY - <<'EOF'
import json, sys
d = json.load(open("runs/v7r/p4_joint/p4_decision.json"))
sys.exit(0 if d.get("second_seed") else 2)
EOF
  then
    local sn
    sn=$($PY -c "import json;print(json.load(open('$RUNS/p4_joint/p4_decision.json'))['second_seed']['arm'])")
    run_cmd e4 seed43 $PY -m tools.v7r.p2_train --name "${sn}_s43" \
      --mode corrector --seed 43 --ckpt-steps 0,1500,3000
    for ck in "$RUNS"/p2_predictor/${sn}_s43/step_*.pt; do
      [[ -e "$ck" ]] || continue
      s=$(basename "$ck" .pt)
      eval_arm "runs/v7r/p2_predictor/${sn}_s43/eval_${s}" "NPred" \
        "runs/v7r/p2_predictor/${sn}_s43/${s}.pt" e4
    done
    run_cmd e4 gate $PY -m tools.v7r.gates p2
  fi
  mark_done e4 runs/v7r/p4_joint/p4_decision.json
}

stage_p3() {
  stage_done p3 && return 0
  gpu_free || fail "GPU busy"
  eval_arm "runs/v7r/p3_corrector/R0" "R0" "$B4CKPT" p3
  eval_arm "runs/v7r/p3_corrector/R1c" "R1c" "$B4CKPT" p3
  eval_arm "runs/v7r/p3_corrector/R2c" "R2c" "$B4CKPT" p3
  if p1_local_go; then
    eval_arm "runs/v7r/p3_corrector/R3c" "R3c" "$B4CKPT" p3
    eval_arm "runs/v7r/p3_corrector/R4c" "R4c" "$B4CKPT" p3
  else
    log "[p3] P1 NO_LOCAL -> short R1 paths instead of R3/R4 (predefined branch)"
    eval_arm "runs/v7r/p3_corrector/R1c_s1" "R1c_s1" "$B4CKPT" p3
    eval_arm "runs/v7r/p3_corrector/R1c_s2" "R1c_s2" "$B4CKPT" p3
  fi
  run_cmd p3 gate $PY -m tools.v7r.gates p3
  # R5c: best predictor + best available corrector
  if $PY - <<'EOF'
import json, sys
try:
    v = json.load(open("runs/v7r/p2_predictor/p2_gate.json"))
except Exception:
    sys.exit(2)
sys.exit(0 if v.get("best_ckpt") else 2)
EOF
  then
    CK=$(best_corrector_ckpt)
    eval_arm "runs/v7r/p3_corrector/R5c" "R5c" "$CK" p3
  else
    [[ -f "$REPO/runs/v7r/p3_corrector/R5c/SKIPPED_BY_GATE.json" ]] || \
      skip_gate "runs/v7r/p3_corrector/R5c" "SKIPPED_BY_GATE" "requires a trained predictor"
  fi
  # P3-C content-constraint variants only when a real-distribution corrector
  # beats B4 DINO-S at B4-level content
  if p1_local_go && $PY - <<'EOF'
import json, sys
try:
    g = json.load(open("runs/v7r/p3_corrector/p3_gate.json"))
    b4 = json.load(open("runs/v6_r2/eval_B4_distill50/eval.json"))["cross600"]
except Exception:
    sys.exit(2)
a = g["arms"].get(g.get("best_corrector") or "", None)
sys.exit(0 if a and a["dino_s"] > b4["dino_s"] else 2)
EOF
  then
    eval_arm "runs/v7r/p3_corrector/R3c_noC" "R3c_noC" "$B4CKPT" p3
    eval_arm "runs/v7r/p3_corrector/R3c_disp" "R3c_disp" "$B4CKPT" p3
  fi
  mark_done p3 runs/v7r/p3_corrector/p3_gate.json
}

stage_p4() {
  stage_done p4 && return 0
  gpu_free || fail "GPU busy"
  run_cmd p4 judge $PY -m tools.v7r.p4_decision judge
  mark_done p4 runs/v7r/p4_joint/p4_verdict.json
}

stage_p5() {
  stage_done p5 && return 0
  gpu_free || fail "GPU busy"
  local grid_ckpt="$B4CKPT"
  local grid_arm="R0"
  if [[ -f "$RUNS/p2_predictor/p2_gate.json" ]]; then
    local CK
    CK=$(best_corrector_ckpt)
    if [[ -n "$CK" && -f "$REPO/$CK" ]]; then
      grid_ckpt="$CK"; grid_arm="NPred"
    fi
  fi
  run_cmd p5 grid $PY -m tools.v7r.p5_grid --v7r-arm "$grid_arm" \
    --ckpt "$grid_ckpt"
  run_cmd p5 report $PY -m tools.v7r.p5_report
  mark_done p5 runs/v7r/final_matrix.json runs/v7r/final_verdict.json \
    runs/v7r/visual_grid.png docs/v7/V7R_EXTENSION_REPORT.md
}

log "V7-R+ serial runner start (BUDGET_TOTAL_SEC=${BUDGET_TOTAL_SEC}s)"
stage_p0
stage_p1
stage_p2
stage_e1
stage_e2
stage_e3
stage_e5
stage_e4
stage_p3
stage_p4
stage_p5
eta_update
log "ALL V7-R+ STAGES COMPLETE in $(( $(date +%s) - T0_WALL ))s"
