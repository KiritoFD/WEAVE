#!/usr/bin/env bash
# P0 四项正确性审计（覆盖率 / 缓存一致性 / 反向评分代数 / 频带占比）
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
python3 -u tools/pseudo_gt/p0_audit.py > runs/p2/p0_audit.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date -u '+%F %T')] FAILED p0_audit rc=$rc" >> "$PROG"
  tail -n 30 runs/p2/p0_audit.log
  exit 1
fi
echo "[$(date -u '+%F %T')] p0_audit done -> runs/p2/pgt_p0_audit.json" >> "$PROG"
tail -n 30 runs/p2/p0_audit.log
