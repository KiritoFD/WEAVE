#!/usr/bin/env bash
# Remediation run: gt-build §7 compliant effective batch 256 (microbatch 32 x accum 8),
# step checkpoints 250/500/1000/2000. Three arms -> render -> metrics -> summary.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/pgt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

CELLS="S0_base_b256 S1_positive_b256 S2_reverse_b256"

log "REMEDIATION start: effective batch 256 (32x8) + ckpt 250/500/1000/2000; rerun 3 arms"

for cell in $CELLS; do
  mkdir -p "runs/pgt/$cell"
  python3 -u run.py --config "configs/pgt/$cell.json" > "runs/pgt/$cell/train.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "FAILED train $cell rc=$rc"
    tail -n 15 "runs/pgt/$cell/train.log"
    exit 1
  fi
  log "train $cell done"
done

for cell in $CELLS; do
  ck="runs/pgt/$cell/step_2000.pt"
  if [ ! -f "$ck" ]; then
    ck=$(ls -1v "runs/pgt/$cell"/epoch_*.pt 2>/dev/null | tail -1)
  fi
  if [ -z "$ck" ] || [ ! -f "$ck" ]; then
    log "FAILED render $cell: no checkpoint"
    exit 1
  fi
  python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$ck" --cell "$cell" \
    > "runs/pgt/$cell/render.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "FAILED render $cell rc=$rc"
    tail -n 12 "runs/pgt/$cell/render.log"
    exit 1
  fi
  log "render $cell done ($ck)"
done

for cell in $CELLS; do
  python3 -u tools/12h/p2_metrics.py --gen-cell "$cell" --base-cell A \
    --out "runs/p2/pgt_$cell.json" > "runs/p2/metrics_$cell.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "FAILED metrics $cell rc=$rc"
    tail -n 12 "runs/p2/metrics_$cell.log"
    exit 1
  fi
  log "metrics $cell done"
done

python3 -u _tmp/pgt_summary.py > runs/p2/pgt_summary.md 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "FAILED summary rc=$rc"
  tail -n 20 runs/p2/pgt_summary.md
  exit 1
fi
log "B256 DONE"
tail -n 80 runs/p2/pgt_summary.md
