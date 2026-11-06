#!/usr/bin/env bash
# V4 Stage B: 全量造 E3 / E8 配对缓存（同一基础配对集合），随后 P2 可达性复核
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }
sample() { ( while true; do
    echo "$(date -u +%T) vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) rss_used_mib=$(free -m | awk '/Mem:/{print $3}')"
    sleep 5; done ) > "$1" & echo $!; }
peak() { awk -F'vram_mib=' '{split($2,a," "); if(a[1]+0>vm)vm=a[1]+0} END{print "peak_vram_mib="vm}' "$1"; }
rsspeak() { awk -F'rss_used_mib=' '{v=$2+0; if(v>m)m=v} END{print "peak_rss_used_mib="m}' "$1"; }

log "V4 stageB start"
T0=$(date +%s)
S=$(sample runs/p2/e3_res.log)
python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --spec E3 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/pairs/reverse_e3 \
  > runs/p2/pairs_reverse_e3.log 2>&1
rc=$?; kill $S 2>/dev/null
if [ $rc -ne 0 ]; then log "FAILED E3 rc=$rc"; tail -n 20 runs/p2/pairs_reverse_e3.log; exit 1; fi
log "E3 build done ($(($(date +%s)-T0))s) $(peak runs/p2/e3_res.log) $(rsspeak runs/p2/e3_res.log)"

T1=$(date +%s)
S=$(sample runs/p2/e8_res.log)
python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --spec E8 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/pairs/reverse_e8 \
  > runs/p2/pairs_reverse_e8.log 2>&1
rc=$?; kill $S 2>/dev/null
if [ $rc -ne 0 ]; then log "FAILED E8 rc=$rc"; tail -n 20 runs/p2/pairs_reverse_e8.log; exit 1; fi
log "E8 build done ($(($(date +%s)-T1))s) $(peak runs/p2/e8_res.log) $(rsspeak runs/p2/e8_res.log)"

python3 - <<'PY'
import json, statistics
from pathlib import Path
R=Path('/mnt/i/Latent_Style/pseudo_gt_24h/pairs')
def ids(d):
    m=json.loads((R/d/'manifest.json').read_text()); return {r['sample_id'] for r in m['records']}, m['records']
base,b0=ids('base'); pos,p0=ids('positive'); rev,r0=ids('reverse')
e3,r3=ids('reverse_e3'); e8,r8=ids('reverse_e8')
print('[idset] base=%d positive=%d reverse=%d e3=%d e8=%d' % (len(base),len(pos),len(rev),len(e3),len(e8)))
print('[idset] intersections: base&pos=%d base&rev=%d base&e3=%d base&e8=%d' %
      (len(base&pos),len(base&rev),len(base&e3),len(base&e8)))
for name,recs in (('e3',r3),('e8',r8)):
    cov=[float(r.get('effective_coverage_measured',0) or 0) for r in recs]
    w=[float(r.get('effective_coverage_weighted',0) or 0) for r in recs]
    print('%s: n=%d measured_mean=%.4f weighted_mean=%.4f min=%.4f <0.18=%d' %
          (name,len(recs),statistics.mean(cov),statistics.mean(w),min(cov),sum(1 for c in cov if c<0.18)))
PY
log "stageB builds done; running p2_reach on E3/E8"
T2=$(date +%s)
python3 -u tools/pseudo_gt/p2_reach.py --sets "E3=reverse_e3,S4=reverse_e8" --n 40 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach_e3e8.json > runs/p2/p2_reach_e3e8.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED p2 e3e8 rc=$rc"; tail -n 15 runs/p2/p2_reach_e3e8.log; exit 1; fi
log "p2_reach E3/E8 done ($(($(date +%s)-T2))s)"
log "stageB ALL DONE"
cat /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach_e3e8.json
