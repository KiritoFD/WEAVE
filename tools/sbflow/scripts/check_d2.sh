#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/v11
echo "=== d2 gen/score artifacts ==="
ls -lh "$ROOT"/gen_d2_quick200/manifest.json "$ROOT"/score_d2_quick200/style_scores.json 2>/dev/null || true
echo "=== d2 score log tail ==="
tail -n 6 "$ROOT/logs/d2_score.log" 2>/dev/null || echo "  no log"
echo "=== d2 gen log tail ==="
tail -n 3 "$ROOT/logs/d2_gen.log" 2>/dev/null || echo "  no log"
echo "=== running scorers ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\.(score_cpu|gen5|train5)" | grep -v grep | cut -c1-110 || echo "  none"
