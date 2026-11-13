#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== queue_p2b.log tail ==="
tail -n 14 runs/v6_24h/queue_p2b.log 2>/dev/null || echo "  no log"
echo
echo "=== running process ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -1 | cut -c1-72
echo
echo "=== step progress for the current arm ==="
for a in G1a_rank_weak G1b_rank_strong; do
  f="runs/v6_24h/p2_${a}.log"
  [ -f "$f" ] && { echo "-- $a (old log, ignore if stale)"; grep -a "step [0-9]*/800" "$f" | tail -2; }
done
echo
echo "=== report presence ==="
for a in G0_ctrl G1a_rank_weak G1b_rank_strong; do
  if [ -s "runs/v6/$a/report.json" ]; then
    python3 -c "
import json,sys
d=json.load(open('runs/v6/$a/report.json')); e=d.get('eval200') or {}
pt=e.get('per_target_top1') or {}
t1=sum(pt.values())/len(pt) if pt else 0
print('  $a  dT=%+.4f Top1=%.3f DINO=%.4f LPIPS=%.4f' % (e.get('mean_dT',0), t1, e.get('mean_dino_c',0), e.get('mean_lpips',0)))
"
  else
    echo "  $a  (no report yet)"
  fi
done
