#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
OUT=runs/v4/train_bench.txt
: > "$OUT"

run_one() {  # $1=tag $2=config
  local tag=$1 cfg=$2
  local vf=/tmp/tvram_$tag.txt; : > "$vf"
  local t0=$(date +%s)
  python3 -u run.py --config "$cfg" > "runs/v4/${tag}.log" 2>&1 &
  local pid=$!
  ( while kill -0 $pid 2>/dev/null; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> "$vf"
      sleep 2; done ) &
  local sp=$!
  wait $pid
  local rc=$?
  kill $sp 2>/dev/null
  local el=$(( $(date +%s)-t0 ))
  local pk=$(sort -n "$vf" | tail -1)
  local gs=$(grep -ao 'global_step=[0-9]*' "runs/v4/${tag}.log" | tail -1 | cut -d= -f2)
  local rss=$(grep -aE 'peak|rss' "runs/v4/${tag}.log" | tail -2 | tr '\n' ';' | cut -c1-200)
  echo "tag=$tag rc=$rc elapsed=${el}s global_step=${gs:-NA} peak_vram=${pk}mib $rss" | tee -a "$OUT"
}

run_one bench_mb32 configs/v4/bench_mb32.json
run_one bench_mb64 configs/v4/bench_mb64.json
run_one bench_mb128 configs/v4/bench_mb128.json
echo "TRAIN BENCH DONE" | tee -a "$OUT"
