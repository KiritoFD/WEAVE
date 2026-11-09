#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "now: $(date +%H:%M:%S)"
echo "=== current arm ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -n 1 | cut -c1-90
echo "=== last 3 step lines for each recent arm ==="
for a in S6_style_keys S6b_style_keys48 L8_k1024_dim32 L9_k1024_dim32_lc075; do
  f="runs/v6_24h/train_logs/${a}.log"
  [ -f "$f" ] || continue
  echo "-- $a"
  grep -a "step " "$f" | tail -n 3
done
