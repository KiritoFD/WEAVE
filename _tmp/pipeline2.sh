#!/bin/bash
# 续跑：base 缓存已完成 → 重跑 positive/reverse（批量编码版）→ 冒烟 → 训练 → 渲染 → 指标
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
mkdir -p runs/pgt runs/p2
PROG=runs/pgt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

read -r POS REV < <(python3 - <<'PY'
import json
d = json.load(open('runs/p2/pgt_selection.json'))
print(d['positive']['key'], d['reverse']['key'])
PY
)
POS_M=${POS%_*}; POS_S=${POS#*_}
REV_S=${REV#*_}
log "resume: positive=$POS reverse=$REV"

python3 -u tools/pseudo_gt/build_train_cache.py --arm positive --method "$POS_M" --spec "$POS_S" --versions 2 \
  >> runs/p2/pairs_positive.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "FAILED pairs positive rc=$rc"; exit 1; fi
log "pairs positive done"

python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --method A --spec "$REV_S" --versions 2 \
  >> runs/p2/pairs_reverse.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "FAILED pairs reverse rc=$rc"; exit 1; fi
log "pairs reverse done"

python3 -u run.py --config configs/pgt/S0_smoke.json > runs/pgt/_smoke.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "FAILED smoke rc=$rc"; exit 1; fi
log "smoke done"

for cfg in S0_base S1_positive S2_reverse; do
  mkdir -p "runs/pgt/$cfg"
  python3 -u run.py --config "configs/pgt/$cfg.json" > "runs/pgt/$cfg/train.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "FAILED train $cfg rc=$rc"; exit 1; fi
  log "train $cfg done"
done

for cell in S0_base S1_positive S2_reverse; do
  ck=$(ls -1v "runs/pgt/$cell"/epoch_*.pt 2>/dev/null | tail -1)
  if [ -z "$ck" ]; then log "FAILED render $cell: no checkpoint"; exit 1; fi
  python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$ck" --cell "$cell" \
    > "runs/pgt/$cell/render.log" 2>&1
  rc=$?; if [ $rc -ne 0 ]; then log "FAILED render $cell rc=$rc"; exit 1; fi
  log "render $cell done ($ck)"
done

for cell in S0_base S1_positive S2_reverse; do
  python3 -u tools/12h/p2_metrics.py --gen-cell "$cell" --base-cell A \
    --out "runs/p2/pgt_$cell.json" > "runs/p2/metrics_$cell.log" 2>&1
  rc=$?; if [ $rc -ne 0 ]; then log "FAILED metrics $cell rc=$rc"; exit 1; fi
  log "metrics $cell done"
done

python3 -u _tmp/pgt_summary.py > runs/p2/pgt_summary.md 2>&1
log "ALL DONE"
