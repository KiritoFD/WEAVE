#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/v11
echo "=== processes ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\.(v11_run|train5|gen5|score_cpu)" | grep -v grep | cut -c1-110 || echo "  none"
echo "=== p1 driver log ==="
tail -n 6 "$ROOT/logs/p1.log" 2>/dev/null || echo "  no log"
echo "=== e1 gen/score ==="
ls -lh "$ROOT"/gen_e1_quick200/manifest.json "$ROOT"/score_e1_quick200/style_scores.json 2>/dev/null || echo "  not yet"
echo "=== e1 logs ==="
tail -n 3 "$ROOT/logs/e1_gen.log" 2>/dev/null || echo "  no gen log"
echo "=== e2 ==="
tail -n 3 "$ROOT/logs/e2_train.log" 2>/dev/null || echo "  e2 not started"
