#!/bin/bash
# 一条龙：等待 Q0 → 选臂 → 三份配对缓存 → 冒烟 → 三臂训练 → 三臂渲染 → 三份指标
# 设计原则：任何一步非零退出立即停止并在 PROGRESS.md 记录原因，绝不静默跳过。
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
mkdir -p runs/pgt runs/p2
PROG=runs/pgt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

log "pipeline start"

# 0. 等待 Q0 结束
while pgrep -f "q0_quality.py" >/dev/null 2>&1; do sleep 60; done
if [ ! -s runs/p2/pgt_q0.json ]; then log "HARD BLOCK: runs/p2/pgt_q0.json 缺失"; exit 1; fi
log "Q0 done"

# 1. 选臂
python3 -u tools/pseudo_gt/select.py --q0 runs/p2/pgt_q0.json --out runs/p2/pgt_selection.json \
  >> runs/p2/select.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED select rc=$rc (see runs/p2/select.log)"; exit 1; fi
log "select done"

read -r POS REV < <(python3 - <<'PY'
import json
d = json.load(open('runs/p2/pgt_selection.json'))
print(d['positive']['key'], d['reverse']['key'])
PY
)
log "selection positive=$POS reverse=$REV"
POS_M=${POS%_*}; POS_S=${POS#*_}
REV_M=${REV%_*}; REV_S=${REV#*_}

# 2. 三份配对缓存（断点续跑：ShardWriter 自带 completed_ids）
python3 -u tools/pseudo_gt/build_train_cache.py --arm base --method A --spec "$POS_S" --versions 2 \
  >> runs/p2/pairs_base.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "FAILED pairs base rc=$rc"; exit 1; fi
log "pairs base done"

python3 -u tools/pseudo_gt/build_train_cache.py --arm positive --method "$POS_M" --spec "$POS_S" --versions 2 \
  >> runs/p2/pairs_positive.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "FAILED pairs positive rc=$rc"; exit 1; fi
log "pairs positive done ($POS_M $POS_S)"

python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --method A --spec "$REV_S" --versions 2 \
  >> runs/p2/pairs_reverse.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "FAILED pairs reverse rc=$rc"; exit 1; fi
log "pairs reverse done ($REV_S)"

# 3. 接线冒烟（5 步）
python3 -u run.py --config configs/pgt/S0_smoke.json > runs/pgt/_smoke.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "FAILED smoke rc=$rc (see runs/pgt/_smoke.log)"; exit 1; fi
log "smoke done"

# 4. 三臂训练（内部早停优先，2000 步 cap）
for cfg in S0_base S1_positive S2_reverse; do
  mkdir -p "runs/pgt/$cfg"
  python3 -u run.py --config "configs/pgt/$cfg.json" > "runs/pgt/$cfg/train.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "FAILED train $cfg rc=$rc (see runs/pgt/$cfg/train.log)"; exit 1; fi
  log "train $cfg done"
done

# 5. 三臂渲染 750（复用 P0/P2 协议与 cell A 基线）
for cell in S0_base S1_positive S2_reverse; do
  ck=$(ls -1v "runs/pgt/$cell"/epoch_*.pt 2>/dev/null | tail -1)
  if [ -z "$ck" ]; then log "FAILED render $cell: no checkpoint"; exit 1; fi
  python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$ck" --cell "$cell" \
    > "runs/pgt/$cell/render.log" 2>&1
  rc=$?; if [ $rc -ne 0 ]; then log "FAILED render $cell rc=$rc"; exit 1; fi
  log "render $cell done ($ck)"
done

# 6. 三份指标（600 跨风格 / 150 同风格 / 750 历史口径）
for cell in S0_base S1_positive S2_reverse; do
  python3 -u tools/12h/p2_metrics.py --gen-cell "$cell" --base-cell A \
    --out "runs/p2/pgt_$cell.json" > "runs/p2/metrics_$cell.log" 2>&1
  rc=$?; if [ $rc -ne 0 ]; then log "FAILED metrics $cell rc=$rc"; exit 1; fi
  log "metrics $cell done"
done

python3 -u _tmp/pgt_summary.py > runs/p2/pgt_summary.md 2>&1
log "ALL DONE"
