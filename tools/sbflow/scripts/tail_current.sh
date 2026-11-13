#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "now: $(date +%H:%M:%S)"
for a in L6_highC L7_highC_dim32 L3_core_off; do
  f="runs/v6_24h/train_logs/${a}.log"
  [ -f "$f" ] || continue
  echo "=== $a ==="
  grep -a "step " "$f" | tail -n 2
done
echo "=== current arm ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -n 1 | cut -c1-90
