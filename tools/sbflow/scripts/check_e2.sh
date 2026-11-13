#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/v11
echo "=== e2 score log ==="
tail -n 4 "$ROOT/logs/e2_score.log" 2>/dev/null || echo "  no log"
echo "=== e2 score dir ==="
ls -1 "$ROOT/score_e2_quick200/" 2>/dev/null || echo "  missing"
echo "=== e2 gen ==="
ls -lh "$ROOT/gen_e2_quick200/manifest.json" 2>/dev/null || echo "  missing"
echo "=== running ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\.(v11_run|score_cpu|train5)" | grep -v grep | cut -c1-100 || echo "  none"
echo "=== p1 driver tail ==="
tail -n 4 "$ROOT/logs/p1.log" 2>/dev/null
echo "=== chain log ==="
tail -n 6 "$ROOT/logs/chain.log" 2>/dev/null || echo "  no chain log yet"
