#!/usr/bin/env bash
# Metrics for the three pgt arms + summary (rerun after the p0-root fix).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/pgt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

for cell in S0_base S1_positive S2_reverse; do
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
log "ALL DONE"
cat runs/p2/pgt_summary.md
