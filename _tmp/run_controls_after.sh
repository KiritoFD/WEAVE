#!/usr/bin/env bash
# Wait for the b256 pipeline to release the GPU, then run the 4 missing controls (Q0-level).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/pgt/PROGRESS.md
for i in $(seq 1 240); do
  if tail -n 8 "$PROG" | grep -q 'B256 DONE'; then
    break
  fi
  sleep 60
done
python3 -u tools/pseudo_gt/q0_controls.py > runs/p2/q0_controls.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date -u '+%F %T')] FAILED controls rc=$rc" >> "$PROG"
  tail -n 15 runs/p2/q0_controls.log
  exit 1
fi
echo "[$(date -u '+%F %T')] controls done -> runs/p2/pgt_q0_controls.json" >> "$PROG"
tail -n 30 runs/p2/q0_controls.log
