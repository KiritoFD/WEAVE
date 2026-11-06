#!/usr/bin/env bash
# 小批量基准计时（10 张源图 × 28 配置），用于推算全量耗时
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
python3 -u tools/pseudo_gt/supp_eval.py --limit 10 \
  --out runs/p2/supp_bench.jsonl --meta runs/p2/supp_bench_meta.json \
  > runs/p2/supp_bench.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date -u '+%F %T')] FAILED supp_bench rc=$rc" >> "$PROG"
  tail -n 30 runs/p2/supp_bench.log
  exit 1
fi
echo "[$(date -u '+%F %T')] supp_bench done (10 图)" >> "$PROG"
tail -n 20 runs/p2/supp_bench.log
