#!/usr/bin/env bash
# Wait for controls, then run contact sheet + E two-direction check.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
mkdir -p runs/gt
ln -sf ../pgt/PROGRESS.md runs/gt/PROGRESS.md
PROG=runs/pgt/PROGRESS.md
for i in $(seq 1 180); do
  if tail -n 10 "$PROG" | grep -q 'controls done'; then
    break
  fi
  sleep 60
done
python3 -u tools/pseudo_gt/q0_extra_checks.py > runs/p2/q0_extra.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date -u '+%F %T')] FAILED extras rc=$rc" >> "$PROG"
  tail -n 15 runs/p2/q0_extra.log
  exit 1
fi
echo "[$(date -u '+%F %T')] extras done (contact sheet + E two-direction)" >> "$PROG"
tail -n 6 runs/p2/q0_extra.log
