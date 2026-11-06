#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "--- procs ---"
pgrep -af fix_effective_coverage || echo none
echo "--- logs ---"
ls -la runs/p2/covfix.log runs/p2/stage_covfix.log 2>/dev/null
echo "--- stage log head/tail ---"
head -n 20 runs/p2/stage_covfix.log 2>/dev/null
echo "..."
tail -n 20 runs/p2/stage_covfix.log 2>/dev/null
echo "--- covfix.log tail ---"
tail -n 20 runs/p2/covfix.log 2>/dev/null
echo "--- summary ---"
ls -la runs/p2/coverage_fix_summary.json 2>/dev/null || echo no_summary
tmux ls 2>&1
