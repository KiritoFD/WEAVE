#!/usr/bin/env bash
# P4：Pattern 缓存（R64/F64 × pos/rev，单遍构造双写）→ val 留出集 → nostyle → T1–T5 训练 → 750 渲染 → metrics → FM 探针
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }
PAIRS=$(python3 -c "from tools.pseudo_gt.build_assets import pgt_root; print(pgt_root()/'pairs')")

log "P4 cache build start (R64,F64 × positive/reverse, versions=1, split=train)"
python3 -u tools/pseudo_gt/build_pattern_cache.py --patterns R64,F64 --versions 1 --split train \
  > runs/pgt/build_patterns_train.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED build_patterns_train rc=$rc"; tail -n 30 runs/pgt/build_patterns_train.log; exit 1; fi
log "P4 cache train done: $(tail -n 1 runs/pgt/build_patterns_train.log)"

log "P4 val cache start (limit 400)"
python3 -u tools/pseudo_gt/build_pattern_cache.py --patterns R64,F64 --versions 1 --split val \
  --limit 400 --out-root "$PAIRS/../pairs_val" > runs/pgt/build_patterns_val.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED build_patterns_val rc=$rc"; tail -n 30 runs/pgt/build_patterns_val.log; exit 1; fi
log "P4 val cache done: $(tail -n 1 runs/pgt/build_patterns_val.log)"

log "nostyle cache for T5"
python3 -u tools/pseudo_gt/make_nostyle_cache.py --src pairs/pattern_R64_reverse \
  --dst pairs/pattern_R64_reverse_nostyle > runs/pgt/nostyle.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED nostyle rc=$rc"; tail -n 20 runs/pgt/nostyle.log; exit 1; fi
log "nostyle done: $(tail -n 1 runs/pgt/nostyle.log)"

PV="$PAIRS/../pairs_val"

for CELL in T1_R64_positive T2_F64_positive T3_R64_reverse T4_F64_reverse T5_R64_reverse_nostyle; do
  log "train $CELL start"
  mkdir -p "runs/pgt/$CELL"
  python3 -u run.py --config "configs/pgt/$CELL.json" > "runs/pgt/$CELL/train.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "FAILED train $CELL rc=$rc"; tail -n 20 "runs/pgt/$CELL/train.log"; exit 1; fi
  log "train $CELL done: $(grep -E 'Epoch [0-9]+/' runs/pgt/$CELL/train.log | tail -1 | cut -c1-120)"

  ck="runs/pgt/$CELL/step_000600.pt"
  if [ ! -f "$ck" ]; then ck=$(ls -1v "runs/pgt/$CELL"/step_*.pt 2>/dev/null | tail -1); fi
  if [ -z "${ck:-}" ] || [ ! -f "$ck" ]; then ck=$(ls -1v "runs/pgt/$CELL"/epoch_*.pt 2>/dev/null | tail -1); fi
  if [ -z "${ck:-}" ] || [ ! -f "$ck" ]; then log "FAILED render $CELL: no ckpt"; exit 1; fi
  log "$CELL ckpt=$ck"

  python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$ck" --cell "$CELL" > "runs/pgt/$CELL/render.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "FAILED render $CELL rc=$rc"; tail -n 15 "runs/pgt/$CELL/render.log"; exit 1; fi
  log "render $CELL done: $(tail -n 1 runs/pgt/$CELL/render.log)"

  python3 -u tools/12h/p2_metrics.py --gen-cell "$CELL" --base-cell A \
    --out "runs/p2/pgt_$CELL.json" > "runs/p2/metrics_$CELL.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "FAILED metrics $CELL rc=$rc"; tail -n 15 "runs/p2/metrics_$CELL.log"; exit 1; fi
  log "metrics $CELL done"

  python3 -u tools/pseudo_gt/fm_probe.py --checkpoint "$ck" --arm "$CELL" \
    --probe-cache "R64_pos=$PV/pattern_R64_positive" \
    --probe-cache "F64_pos=$PV/pattern_F64_positive" \
    --probe-cache "R64_rev=$PV/pattern_R64_reverse" \
    --probe-cache "F64_rev=$PV/pattern_F64_reverse" \
    --out "runs/p2/fm_$CELL.json" > "runs/p2/fm_$CELL.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "FAILED fm_probe $CELL rc=$rc"; tail -n 15 "runs/p2/fm_$CELL.log"; exit 1; fi
  log "fm_probe $CELL done"
done
log "STAGE_TRAIN ALL DONE"
