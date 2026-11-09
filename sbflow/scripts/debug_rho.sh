#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== does remote train5.py contain the rho block? ==="
grep -n "rho_swd_over_bridge\|bridge_grad_norm\|loss_before_dist" sbflow/train5.py || echo "  NOT FOUND"
echo "=== the guard line ==="
grep -n "step == start_step + 1" sbflow/train5.py
echo "=== d1 log first lines ==="
head -n 6 runs/blog_sbflow/v11/logs/d1_train.log 2>/dev/null
echo "=== d1 log: any step=3001 line ==="
grep -n "step=3001" runs/blog_sbflow/v11/logs/d1_train.log 2>/dev/null | head -3 || echo "  no step=3001 line"
