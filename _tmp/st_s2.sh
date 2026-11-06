#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
for arm in S1_positive_b256 S2_reverse_b256; do
  echo "========== $arm"
  echo "epoch_files: $(ls -1 runs/pgt/$arm | grep -c '^epoch_')"
  echo "step_files: $(ls -1 runs/pgt/$arm | grep '^step_' | tr '\n' ' ')"
  echo "last_epoch: $(ls -1 runs/pgt/$arm | grep '^epoch_' | sort | tail -n 1)"
  echo "train.log size: $(stat -c %s runs/pgt/$arm/train.log)"
  echo "-- last 3 non-progress lines:"
  grep -v 'it/s\]' runs/pgt/$arm/train.log | tail -n 3 | cut -c1-300
  echo "-- step/lr lines (last 8):"
  grep -E 'step|Step|epoch' runs/pgt/$arm/train.log | grep -v 'it/s\]' | tail -n 8 | cut -c1-300
  echo "-- internal summary:"
  cut -c1-400 runs/pgt/$arm/internal_dynamics_summary.json 2>/dev/null | head -n 8
done
echo "========== gt PROGRESS last 25"
tail -n 25 runs/gt/PROGRESS.md 2>/dev/null | cut -c1-400
