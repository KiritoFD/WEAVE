#!/usr/bin/env bash
# V4 Stage A: P2 可达性（现有缓存）→ positive 补满 26144 → reverse 覆盖率实测 → E8 300 组质检缓存
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
mkdir -p runs/gt runs/p2
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

sample() {  # $1=标签 $2=输出文件
  ( while true; do
      echo "$(date -u +%T) vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) rss_used_mib=$(free -m | awk '/Mem:/{print $3}')"
      sleep 5
    done ) > "$2" &
  echo $!
}

peak() {  # $1=输出文件
  awk -F'vram_mib=' '{split($2,a," "); if(a[1]+0>vm)vm=a[1]+0} END{print "peak_vram_mib="vm}' "$1"
}
rsspeak() { awk -F'rss_used_mib=' '{v=$2+0; if(v>m)m=v} END{print "peak_rss_used_mib="m}' "$1"; }

log "V4 stageA start"
T0=$(date +%s)
python3 -m pytest tests/test_pseudo_gt_masks.py -q > runs/p2/masks_test.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED masks test rc=$rc"; tail -n 20 runs/p2/masks_test.log; exit 1; fi
log "masks test PASS"
SAMP=$(sample p2 runs/p2/p2_reach_res.log)
python3 -u tools/pseudo_gt/p2_reach.py \
  --sets "A_P5=positive,E7=reverse" --n 40 --out /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach.json \
  > runs/p2/p2_reach.log 2>&1
rc=$?
kill $SAMP 2>/dev/null
if [ $rc -ne 0 ]; then log "FAILED p2_reach rc=$rc"; tail -n 20 runs/p2/p2_reach.log; exit 1; fi
log "p2_reach done ($(($(date +%s)-T0))s) $(peak runs/p2/p2_reach_res.log) $(rsspeak runs/p2/p2_reach_res.log) samples=80"

T1=$(date +%s)
python3 -u tools/pseudo_gt/build_train_cache.py --arm positive --method A --spec P5 \
  >> runs/p2/pairs_positive_resume.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED positive resume rc=$rc"; tail -n 15 runs/p2/pairs_positive_resume.log; exit 1; fi
log "positive resume done ($(($(date +%s)-T1))s) -> $(python3 -c "import json;print(json.load(open('/mnt/i/Latent_Style/pseudo_gt_24h/pairs/positive/manifest.json'))['count'])") records"

T2=$(date +%s)
python3 -u tools/pseudo_gt/fix_effective_coverage.py --arms reverse \
  >> runs/p2/covfix_reverse.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "FAILED covfix reverse rc=$rc"; tail -n 15 runs/p2/covfix_reverse.log; exit 1; fi
log "covfix reverse done ($(($(date +%s)-T2))s)"

T3=$(date +%s)
SAMP=$(sample e8 runs/p2/e8_preview_res.log)
python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --spec E8 --limit 150 \
  --out /mnt/i/Latent_Style/pseudo_gt_24h/pairs/reverse_e8 \
  > runs/p2/pairs_reverse_e8_preview.log 2>&1
rc=$?
kill $SAMP 2>/dev/null
if [ $rc -ne 0 ]; then log "FAILED e8 preview rc=$rc"; tail -n 15 runs/p2/pairs_reverse_e8_preview.log; exit 1; fi
log "E8 preview 300 done ($(($(date +%s)-T3))s) $(peak runs/p2/e8_preview_res.log) $(rsspeak runs/p2/e8_preview_res.log)"

log "stageA ALL DONE total $(($(date +%s)-T0))s"
echo "=== p2_reach ==="; cat /mnt/i/Latent_Style/pseudo_gt_24h/p2_reach.json
echo "=== covfix reverse ==="; cat /mnt/i/Latent_Style/pseudo_gt_24h/pairs/coverage_fix_summary.json
echo "=== e8 preview tail ==="; tail -n 5 runs/p2/pairs_reverse_e8_preview.log
