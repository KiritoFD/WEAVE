#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== P2 progress ==="
for a in G0_ctrl G1a_rank_weak G1b_rank_strong; do
  f="runs/v6_24h/p2_${a}.log"
  if [ -f "$f" ]; then
    echo "-- $a"
    grep -aE "G1 diag step1|init_from|rank_critic|step (1|200|400|600|800)/|Eval200" "$f" | tail -n 6
  else
    echo "-- $a  (not started)"
  fi
done
echo
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -1 | cut -c1-70
