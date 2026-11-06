#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
CKPT=runs/v4/S1_A_P5/step_000600.pt
RR=/mnt/i/Latent_Style/pseudo_gt_24h/bench_render
OUT=runs/v4/batch_bench2.txt
: > "$OUT"

run_one() {  # $1=batch $2=metric_batch $3=threads $4=tag
  local B=$1 MB=$2 TH=$3 tag=$4
  local vf=/tmp/vram_$tag.txt; : > "$vf"
  local t0=$(date +%s)
  python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$CKPT" --cell "$tag" \
    --render-root "$RR" --max-src-samples 30 --num-steps 8 --enable-clip \
    --batch "$B" --decode-batch 8 --metric-batch "$MB" --save-threads "$TH" > "runs/v4/${tag}.log" 2>&1 &
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
  local wall=$(grep -a '\[pgt_r\] done' "runs/v4/${tag}.log" | sed 's/.*"wall_sec_render": /wall=/; s/,.*//')
  local ips=$(awk -v w="${wall#wall=}" 'BEGIN{ if (w+0>0) printf "%.2f", 750/(w+0); else print "nan" }')
  echo "batch=$B metric_batch=$MB save_threads=$TH tag=$tag rc=$rc total=${el}s ${wall} imgs_per_s=$ips peak_vram=${pk}mib" | tee -a "$OUT"
}

run_one 64 32 8  b64_t8
run_one 96 32 8  b96_t8
run_one 64 128 8 b64_m128_t8
echo "BENCH2 DONE" | tee -a "$OUT"
