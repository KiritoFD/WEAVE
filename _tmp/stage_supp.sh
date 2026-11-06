#!/usr/bin/env bash
# P1/P2/P3 全量逐样本记录：pool1(seed2024,28 配置) + pool2(2025,14) + pool3(2026,14)
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
KEY="R64,G64,C64,S64,F64,R128,G128,C128,A_P5,CTRL_none,CTRL_rand,CTRL_same,CTRL_destyle,CTRL_seed"

echo "[$(date -u '+%F %T')] supp_full start (300 图 × 28 配置)" >> "$PROG"
python3 -u tools/pseudo_gt/supp_eval.py --seed 2024 \
  --out runs/p2/supp_records.jsonl --meta runs/p2/supp_meta.json \
  > runs/p2/supp_full.log 2>&1 || { echo "[$(date -u '+%F %T')] FAILED supp_full" >> "$PROG"; tail -n 40 runs/p2/supp_full.log; exit 1; }
echo "[$(date -u '+%F %T')] supp_full done: $(grep -c '' runs/p2/supp_records.jsonl) rows, $(python3 -c 'import json;print(json.load(open("runs/p2/supp_meta.json"))["total_seconds"])')s" >> "$PROG"

echo "[$(date -u '+%F %T')] supp_pool2 start (seed 2025)" >> "$PROG"
python3 -u tools/pseudo_gt/supp_eval.py --seed 2025 --configs "$KEY" \
  --out runs/p2/supp_pool2.jsonl --meta runs/p2/supp_pool2_meta.json \
  > runs/p2/supp_pool2.log 2>&1 || { echo "[$(date -u '+%F %T')] FAILED supp_pool2" >> "$PROG"; tail -n 40 runs/p2/supp_pool2.log; exit 1; }
echo "[$(date -u '+%F %T')] supp_pool2 done: $(grep -c '' runs/p2/supp_pool2.jsonl) rows" >> "$PROG"

echo "[$(date -u '+%F %T')] supp_pool3 start (seed 2026)" >> "$PROG"
python3 -u tools/pseudo_gt/supp_eval.py --seed 2026 --configs "$KEY" \
  --out runs/p2/supp_pool3.jsonl --meta runs/p2/supp_pool3_meta.json \
  > runs/p2/supp_pool3.log 2>&1 || { echo "[$(date -u '+%F %T')] FAILED supp_pool3" >> "$PROG"; tail -n 40 runs/p2/supp_pool3.log; exit 1; }
echo "[$(date -u '+%F %T')] supp_pool3 done: $(grep -c '' runs/p2/supp_pool3.jsonl) rows" >> "$PROG"
echo "[$(date -u '+%F %T')] stage_supp ALL DONE" >> "$PROG"
