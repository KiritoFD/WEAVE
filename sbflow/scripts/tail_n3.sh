#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
for a in N3_dim32 N4_steps400 N5_inv_off; do
  f="runs/v6_24h/train_logs/${a}.log"
  [ -f "$f" ] || continue
  echo "=== $a ==="
  grep -a "step " "$f" | tail -n 2
  grep -a "Eval200" "$f" | tail -n 1 || true
done
