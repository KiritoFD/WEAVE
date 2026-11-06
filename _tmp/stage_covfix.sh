#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
if pgrep -f 'tools/pseudo_gt' > /dev/null; then echo "ABORT: pgt proc alive"; exit 1; fi
free -g | head -2
T0=$(date +%s)
python3 -u tools/pseudo_gt/fix_effective_coverage.py 2>&1 | tee runs/p2/covfix.log
rc=${PIPESTATUS[0]}
echo "[$(date -u '+%F %T')] fix_effective_coverage rc=$rc elapsed=$(( $(date +%s) - T0 ))s" >> runs/gt/PROGRESS.md
cat runs/p2/coverage_fix_summary.json 2>/dev/null
