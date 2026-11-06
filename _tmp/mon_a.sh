#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "--- procs ---"; pgrep -af 'p2_reach|build_train_cache|fix_effective_coverage' || echo none
echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- progress tail ---"; tail -n 8 runs/gt/PROGRESS.md | cut -c1-260
echo "--- p2_reach.log ---"; tail -n 25 runs/p2/p2_reach.log 2>/dev/null | cut -c1-300
echo "--- positive resume ---"; tail -n 3 runs/p2/pairs_positive_resume.log 2>/dev/null | cut -c1-200
echo "--- e8 preview ---"; tail -n 4 runs/p2/pairs_reverse_e8_preview.log 2>/dev/null | cut -c1-200
echo "--- covfix reverse ---"; tail -n 6 runs/p2/covfix_reverse.log 2>/dev/null | cut -c1-250
echo "--- stage_a2 out ---"; tail -n 6 runs/p2/stage_a2.out 2>/dev/null | cut -c1-250
