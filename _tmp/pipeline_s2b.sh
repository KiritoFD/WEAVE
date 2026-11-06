#!/usr/bin/env bash
# S2b：真正的第三臂。E6 = 羽化 + 128+12×64（与正向 A_P5 硬贴 4×128 在像素上不同）。
# 等 seed2025 复核结束后再启动，避免插队打断 b256。
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/pgt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

CELL=S2_reverse_e6_b256
for i in $(seq 1 300); do
  if tail -n 12 "$PROG" | grep -q 'seed2025 done'; then break; fi
  if tail -n 6 "$PROG" | grep -q 'FAILED seed2025'; then
    log "FAILED upstream seed2025; S2b aborted"; exit 1
  fi
  sleep 60
done

OUT=$(python3 -c "from paths import asset_root; print(asset_root()/'pseudo_gt_24h'/'pairs'/'reverse_e6')")
mkdir -p "$(dirname "$OUT")"
log "S2b start: build reverse_e6 cache (E6 feathered 128+12x64)"
python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --method A --spec E6 \
  --versions 2 --out "$OUT" > runs/pgt/build_reverse_e6.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED cache reverse_e6 rc=$rc"; tail -n 15 runs/pgt/build_reverse_e6.log; exit 1; fi
log "cache reverse_e6 done"

mkdir -p "runs/pgt/$CELL"
python3 -u run.py --config "configs/pgt/$CELL.json" > "runs/pgt/$CELL/train.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED train $CELL rc=$rc"; tail -n 15 "runs/pgt/$CELL/train.log"; exit 1; fi
log "train $CELL done"

ck="runs/pgt/$CELL/step_2000.pt"
if [ ! -f "$ck" ]; then ck=$(ls -1v "runs/pgt/$CELL"/epoch_*.pt 2>/dev/null | tail -1); fi
if [ -z "${ck:-}" ] || [ ! -f "$ck" ]; then log "FAILED render $CELL: no ckpt"; exit 1; fi
python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$ck" --cell "$CELL" > "runs/pgt/$CELL/render.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED render $CELL rc=$rc"; tail -n 12 "runs/pgt/$CELL/render.log"; exit 1; fi
log "render $CELL done ($ck)"

python3 -u tools/12h/p2_metrics.py --gen-cell "$CELL" --base-cell A \
  --out "runs/p2/pgt_$CELL.json" > "runs/p2/metrics_$CELL.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED metrics $CELL rc=$rc"; tail -n 12 "runs/p2/metrics_$CELL.log"; exit 1; fi
log "metrics $CELL done"

python3 -u _tmp/pgt_summary.py > runs/p2/pgt_summary.md 2>&1 || true
log "S2B DONE"
tail -n 60 runs/p2/pgt_summary.md
