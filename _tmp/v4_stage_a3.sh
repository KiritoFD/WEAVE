#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

python3 - <<'PY'
import json
from pathlib import Path
R=Path('/mnt/i/Latent_Style/pseudo_gt_24h/pairs')
def pairs(d):
    m=json.loads((R/d/'manifest.json').read_text())
    return {(r['source_id'],r['donor_id']) for r in m['records']}, m['records']
b,b0=pairs('base'); p,p0=pairs('positive'); r,r0=pairs('reverse')
print('[pairset] base=%d positive=%d reverse=%d | b&p=%d b&r=%d p&r=%d' %
      (len(b),len(p),len(r),len(b&p),len(b&r),len(p&r)))
print('[pairset] only_in_base_vs_positive=%d' % len(set(b0[i]['sample_id'] for i in range(len(b0)))))
PY

python3 - <<'PY'
import json, statistics
m=json.loads(open('/mnt/i/Latent_Style/pseudo_gt_24h/pairs/coverage_fix_summary.json').read())
for arm,s in m.items():
    if arm in ('base','positive','reverse'):
        print('[covfix]', arm, 'measured_n=%d mean=%.4f orig=%.4f frac_drop=%.3f' %
              (s['measured_n'], s['coverage_measured_mean'], s['coverage_original_mean'], s['frac_images_with_drop']))
PY

T=$(date +%s)
python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --spec E8 --limit 150 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/pairs/reverse_e8 \
  >> runs/p2/pairs_reverse_e8_preview.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED e8 preview rc=$rc"; tail -n 12 runs/p2/pairs_reverse_e8_preview.log; exit 1; fi
log "E8 preview done ($(($(date +%s)-T))s)"

python3 -u tools/pseudo_gt/p2_reach.py --sets "S4_preview=reverse_e8" --n 40 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach_e8.json > runs/p2/p2_reach_e8.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED p2 e8 rc=$rc"; tail -n 12 runs/p2/p2_reach_e8.log; exit 1; fi
log "p2_reach E8 preview done"
cat /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach_e8.json
