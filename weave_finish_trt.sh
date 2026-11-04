#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/v2/run_v2_v3_750.log
mkdir -p runs/v2/logs runs/v2/eval
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 等 V2 训练自然跑满 1600 步并落盘，然后停掉旧编排（其评测走 torch decoder，非锚点口径）
while [ ! -f runs/v2/V2/step_001600.pt ]; do sleep 5; done
sleep 2
pgrep -f "weave_run[_]v2_v3" | xargs -r kill -9
pgrep -f "eval[_]fast" | xargs -r kill -9
pgrep -f "python3 -m v2[.]train" | xargs -r kill -9
sleep 3
log "=== V2 1600 ckpt ready; old orchestrator stopped ==="

# 与锚点完全同一路径：TRT-256-fp16, eval_batch 8, decode-batch 8, 8 steps, 无图
run_eval() {
  local ck=$1
  local variant=$2
  local step=$3
  local out="runs/v2/eval/${variant}_step_${step}_750"
  if [ ! -f "$ck" ]; then log "SKIP eval (missing $ck)"; return; fi
  rm -rf "$out"
  log "=== EVAL $variant $step start ==="
  python3 tools/12h/eval_fast_trt_fp16.py --protocol paper --checkpoint "$ck" \
    --output "$out" --device cuda --batch 8 --metric-batch 32 --decode-batch 8 --workers 4 \
    --no-save-images --gen-dtype fp32 --vae-dtype fp32 --sync-timing --decoder trt \
    >> "runs/v2/logs/eval_${variant}_${step}.log" 2>&1
  local rc=$?
  log "=== EVAL $variant $step exit=$rc ==="
  if [ -f "$out/eval_fast_meta.json" ]; then
    log "TIMING $variant $step $(grep -o '\"wall_total\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"generation\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"metric_clip\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"metric_lpips\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"metric_dino\": [0-9.]*' "$out/eval_fast_meta.json") $(grep -o '\"vae_decode\": [0-9.]*' "$out/eval_fast_meta.json")"
  fi
}

run_eval runs/v2/V2/step_000500.pt V2 000500
run_eval runs/v2/V2/step_001000.pt V2 001000
run_eval runs/v2/V2/step_001600.pt V2 001600

log "=== TRAIN V3 start ==="
python3 -m v2.train --config configs/v2/V3.json --variant V3 \
  --steps 1600 --batch-size 144 --accumulation-steps 1 --save-at 500,1000,1600 --diag-every 250 \
  >> runs/v2/logs/train_V3.log 2>&1
rc=$?
log "=== TRAIN V3 exit=$rc ==="

run_eval runs/v2/V3/step_000500.pt V3 000500
run_eval runs/v2/V3/step_001000.pt V3 001000
run_eval runs/v2/V3/step_001600.pt V3 001600

log "=== ALL DONE ==="
