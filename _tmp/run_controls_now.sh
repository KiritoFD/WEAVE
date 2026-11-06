#!/usr/bin/env bash
# Run the 4 Q0 controls now (b256 already finished); write marker on success.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/pgt/PROGRESS.md
python3 -u tools/pseudo_gt/q0_controls.py > runs/p2/q0_controls.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date -u '+%F %T')] FAILED controls rc=$rc" >> "$PROG"
  tail -n 25 runs/p2/q0_controls.log
  exit 1
fi
echo "[$(date -u '+%F %T')] controls done -> runs/p2/pgt_q0_controls.json" >> "$PROG"
tail -n 20 runs/p2/q0_controls.log
