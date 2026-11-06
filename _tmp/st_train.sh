#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
for d in S0_base_b256 S1_positive_b256 S2_reverse_b256; do
  echo "=== $d"
  cat "runs/pgt/$d/internal_dynamics_summary.json" 2>/dev/null
  echo "-- train.log tail"
  tail -n 12 "runs/pgt/$d/train.log" 2>/dev/null
  echo "-- last epoch files"
  ls -1 "runs/pgt/$d" | tail -5
done
echo "=== configs ==="
for c in configs/pgt/T1_R64_positive.json configs/pgt/T2_F64_positive.json configs/pgt/T3_R64_reverse.json configs/pgt/T4_F64_reverse.json configs/pgt/T5_R64_reverse_nostyle.json; do echo "-- $c"; cat "$c"; done
