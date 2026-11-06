#!/usr/bin/env bash
# 等 extras 完成后，做跨 seed 复核（seed=2025，10 组候选）。
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/pgt/PROGRESS.md
for i in $(seq 1 240); do
  if tail -n 10 "$PROG" | grep -q 'extras done'; then
    break
  fi
  sleep 60
done
python3 -u tools/pseudo_gt/q0_quality.py \
  --seed 2025 --n-per-style 60 \
  --configs A_P5,A_P2,A_P7,A_P9,B_P9,E7,E1,E3,E6,D_P7 \
  --out runs/p2/pgt_q0_seed2025.json > runs/p2/q0_seed2025.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date -u '+%F %T')] FAILED seed2025 rc=$rc" >> "$PROG"
  tail -n 20 runs/p2/q0_seed2025.log
  exit 1
fi
echo "[$(date -u '+%F %T')] seed2025 done (10 configs, seed=2025)" >> "$PROG"
tail -n 6 runs/p2/q0_seed2025.log
