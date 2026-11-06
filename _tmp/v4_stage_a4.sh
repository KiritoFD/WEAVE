#!/usr/bin/env bash
# V4 Stage A4: E3 / E1（16×64 25% feather）300 组预览 + P2 可达性，决定 S3/S4a 是否建全量
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

for spec in E3 E1; do
  T=$(date +%s)
  python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --spec $spec --limit 150 \
    --out /mnt/i/Latent_Style/pseudo_gt_24h/pairs/reverse_$(echo $spec | tr 'A-Z' 'a-z') \
    > runs/p2/pairs_reverse_${spec}_preview.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "FAILED $spec preview rc=$rc"; tail -n 12 runs/p2/pairs_reverse_${spec}_preview.log; exit 1; fi
  log "$spec preview done ($(($(date +%s)-T))s)"
done

T=$(date +%s)
python3 -u tools/pseudo_gt/p2_reach.py --sets "E3_preview=reverse_e3,E1_preview=reverse_e1" --n 40 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach_e3e1.json > runs/p2/p2_reach_e3e1.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED p2 e3e1 rc=$rc"; tail -n 12 runs/p2/p2_reach_e3e1.log; exit 1; fi
log "p2_reach E3/E1 preview done ($(($(date +%s)-T))s)"
cat /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach_e3e1.json
