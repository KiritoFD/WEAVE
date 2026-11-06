#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
date -u
echo "== PROGRESS =="
tail -n 14 runs/pgt/PROGRESS.md 2>/dev/null
echo "== PY PROCS =="
ps -ef | grep -E "python3? " | grep -v grep | awk '{print $2, $8, $9, $10, $11, $12}' | head -6
echo "== GPU =="
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "== Q0 LOG =="
tail -n 3 runs/p2/pgt_q0.log 2>/dev/null
for f in runs/p2/pairs_base.log runs/p2/pairs_positive.log runs/p2/pairs_reverse.log; do
  [ -f "$f" ] && { echo "--- $f"; tail -n 2 "$f"; }
done
for f in runs/pgt/_smoke.log runs/pgt/S0_base/train.log runs/pgt/S1_positive/train.log runs/pgt/S2_reverse/train.log; do
  [ -f "$f" ] && { echo "--- $f"; tail -n 3 "$f"; }
done
for f in runs/pgt/S0_base/render.log runs/pgt/S1_positive/render.log runs/pgt/S2_reverse/render.log; do
  [ -f "$f" ] && { echo "--- $f"; tail -n 2 "$f"; }
done
echo "== ARTIFACTS =="
ls -la runs/p2/pgt_q0.json runs/p2/pgt_selection.json runs/p2/pgt_*.json 2>/dev/null
du -sh /mnt/i/Latent_Style/pseudo_gt_24h/pairs/* 2>/dev/null
ls -la runs/pgt/*/internal_dynamics_summary.json 2>/dev/null
echo "== DISK =="
df -h /mnt/i | tail -1
