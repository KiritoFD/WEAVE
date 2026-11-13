#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== style-target caches ==="
ls -la runs/v6/cache/ | grep -i indomain || echo "  none"
echo
echo "=== L4 log first lines ==="
head -n 12 runs/v6_24h/train_logs/L4_ref_full.log 2>/dev/null | grep -aE "reference|params|B1" || echo "  (no match)"
echo
echo "=== current arm elapsed ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -n 1 | cut -c1-80
echo "now: $(date +%H:%M:%S)"
