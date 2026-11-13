#!/usr/bin/env bash
set -uo pipefail
pkill -9 -f "sbflow.score_cpu" 2>/dev/null || true
sleep 1
rm -rf /mnt/i/Latent_Style/WEAVE/runs/blog_sbflow/v10/score_04_full600
echo "cleaned partial scoring"
ps -eo pid,args --no-headers | grep -E "sbflow\." | grep -v grep | cut -c1-90 || echo "no sbflow procs"
