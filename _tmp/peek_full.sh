#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
date -u '+%F %T UTC'
echo "--- tmux ---"
tmux ls 2>&1 | head -5
echo "--- rows ---"
for f in runs/p2/supp_records.jsonl runs/p2/supp_pool2.jsonl runs/p2/supp_pool3.jsonl; do
  [ -f "$f" ] && echo "$f $(grep -c '' "$f")"
done
echo "--- log tail ---"
tail -n 6 runs/p2/supp_full.log 2>&1
echo "--- progress ---"
tail -n 8 runs/gt/PROGRESS.md 2>&1
echo "--- procs ---"
pgrep -af supp_eval | head -5
echo "--- mem ---"
free -g | head -3
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
