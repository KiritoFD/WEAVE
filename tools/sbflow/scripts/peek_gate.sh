#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
G=runs/blog_sbflow/v10/gen_04_full600
echo "=== generated PNGs ==="
ls "$G/generated" 2>/dev/null | wc -l
echo "=== gen log tail ==="
tail -n 6 runs/blog_sbflow/v10/logs/04_gen.log 2>/dev/null || echo "  (none)"
echo "=== score log tail ==="
tail -n 4 runs/blog_sbflow/v10/logs/04_score.log 2>/dev/null || echo "  (none)"
echo "=== processes ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\." | grep -v grep | cut -c1-110
