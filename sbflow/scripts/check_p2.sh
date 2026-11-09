#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== v11 tree ==="
ls -1 runs/blog_sbflow/v11/ 2>/dev/null || echo "  none"
echo "=== p2 log tail ==="
tail -n 12 runs/blog_sbflow/v11/logs/p2.log 2>/dev/null || echo "  no log"
echo "=== p2 gen logs ==="
for f in runs/blog_sbflow/v11/logs/p2_*_gen.log; do
  [ -f "$f" ] && { echo "  -- $f"; tail -n 2 "$f"; }
done
echo "=== processes ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\.(p2_edm_control|gen5|score_cpu)" | grep -v grep | cut -c1-120 || echo "  none"
