#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }
pkill -f build_train_cache && log "stopped no-op positive resume"
sleep 2

python3 - <<'PY'
import json
from pathlib import Path
R=Path('/mnt/i/Latent_Style/pseudo_gt_24h/pairs')
def ids(d):
    m=json.loads((R/d/'manifest.json').read_text())
    return {r['sample_id'] for r in m['records']}
a,b,c=ids('base'),ids('positive'),ids('reverse')
print('[idset] base/%d positive/%d reverse/%d | base&positive=%d base&reverse=%d pos&rev=%d' %
      (len(a),len(b),len(c),len(a&b),len(a&c),len(b&c)))
PY
log "idset checked"

T=$(date +%s)
python3 -u tools/pseudo_gt/fix_effective_coverage.py --arms reverse \
  >> runs/p2/covfix_reverse.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED covfix reverse rc=$rc"; tail -n 15 runs/p2/covfix_reverse.log; exit 1; fi
log "covfix reverse done ($(($(date +%s)-T))s)"

T=$(date +%s)
python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --spec E8 --limit 150 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/pairs/reverse_e8 \
  > runs/p2/pairs_reverse_e8_preview.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED e8 preview rc=$rc"; tail -n 20 runs/p2/pairs_reverse_e8_preview.log; exit 1; fi
log "E8 preview done ($(($(date +%s)-T))s)"
tail -n 6 runs/p2/pairs_reverse_e8_preview.log | cut -c1-220
python3 -u tools/pseudo_gt/p2_reach.py --sets "S4_preview=reverse_e8" --n 40 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach_e8.json > runs/p2/p2_reach_e8.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED p2 e8 rc=$rc"; tail -n 15 runs/p2/p2_reach_e8.log; exit 1; fi
log "p2_reach E8 preview done"
cat /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach_e8.json
