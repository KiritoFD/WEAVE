#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "=== time: $(date '+%F %T')"
echo "--- procs ---"
pgrep -a -f "supp_eval|p0_audit|build_pattern|train.py|render" | head -20
echo "--- mem ---"
free -g | head -3
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- supp_bench.log ---"
tail -n 25 runs/p2/supp_bench.log 2>/dev/null
echo "--- bench jsonl lines ---"
wc -l runs/p2/supp_bench.jsonl 2>/dev/null
echo "--- bench meta ---"
cat runs/p2/supp_bench_meta.json 2>/dev/null
echo "--- p0 audit ---"
cat runs/p2/pgt_p0_audit.json 2>/dev/null | head -c 200
