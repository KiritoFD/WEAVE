#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
export HF_HOME="${WEAVE_EVAL_CACHE_ROOT:-$ROOT/../eval_cache}/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/v2/run_v2_v3_750.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 等主流程（V3 训练 + 其评测）退出，避免评测与训练抢卡
for i in $(seq 1 1080); do
  pgrep -f "weave_finish[_]trt" >/dev/null || break
  sleep 5
done
log "=== backfill: main orchestrator gone, checking all ckpts ==="

run_eval() {
  local ck=$1
  local variant=$2
  local step=$3
  local out="runs/v2/eval/${variant}_step_${step}_750"
  [ -f "$ck" ] || { log "backfill SKIP missing $ck"; return; }
  [ -f "$out/eval_fast_meta.json" ] && { log "backfill HAVE ${variant} ${step}"; return; }
  rm -rf "$out"
  log "=== backfill EVAL $variant $step start ==="
  python3 tools/12h/eval_fast_trt_fp16.py --protocol paper --checkpoint "$ck" \
    --output "$out" --device cuda --batch 8 --metric-batch 32 --decode-batch 8 --workers 4 \
    --no-save-images --gen-dtype fp32 --vae-dtype fp32 --sync-timing --decoder trt \
    >> "runs/v2/logs/eval_${variant}_${step}.log" 2>&1
  local rc=$?
  log "=== backfill EVAL $variant $step exit=$rc ==="
  if [ -f "$out/eval_fast_meta.json" ]; then
    log "TIMING $variant $step $(grep -o '\"wall_total\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"generation\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"vae_decode\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"metric_clip\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"metric_lpips\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"metric_dino\": [0-9.]*' "$out/eval_fast_meta.json")"
  fi
}

for variant in V2 V3; do
  for step in 000500 001000 001600; do
    run_eval "runs/v2/${variant}/step_${step}.pt" "$variant" "$step"
  done
done
log "=== backfill ALL DONE ==="
