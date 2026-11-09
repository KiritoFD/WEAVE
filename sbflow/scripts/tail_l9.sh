#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "now: $(date +%H:%M:%S)"
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -n 1 | cut -c1-80
echo "--- last steps ---"
for a in L9_k1024_dim32_lc075 L11_k1024_dim16; do
  f="runs/v6_24h/train_logs/${a}.log"
  [ -f "$f" ] || continue
  echo "== $a"
  grep -a "step " "$f" | tail -n 2
done
