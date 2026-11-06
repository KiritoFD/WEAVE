#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "now $(date '+%F %T')"
pgrep -a -f supp_eval | head -3
tail -n 6 runs/p2/supp_full.log 2>/dev/null
tail -n 6 runs/p2/supp_pool2.log 2>/dev/null
tail -n 6 runs/p2/supp_pool3.log 2>/dev/null
echo "rows full=$(grep -c '' runs/p2/supp_records.jsonl 2>/dev/null) pool2=$(grep -c '' runs/p2/supp_pool2.jsonl 2>/dev/null) pool3=$(grep -c '' runs/p2/supp_pool3.jsonl 2>/dev/null)"
tail -n 8 runs/gt/PROGRESS.md
free -g | head -2
