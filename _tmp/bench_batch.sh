#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
CKPT=$(ls -t runs/v4/S1_A_P5/step_000600.pt | head -n 1)
RR=/mnt/i/Latent_Style/pseudo_gt_24h/bench_render
mkdir -p "$RR"
OUT=runs/v4/batch_bench.txt
: > "$OUT"

sample_vram() {  # $1=file $2=pid
  while kill -0 "$2" 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> "$1"
    sleep 2
  done
}

run_one() {  # $1=batch $2=decode_batch $3=tag
  local B=$1 DB=$2 tag=$3
  local vf=/tmp/vram_$tag.txt
  : > "$vf"
  local t0=$(date +%s)
  python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$CKPT" --cell "$tag" \
    --render-root "$RR" --max-src-samples 30 --num-steps 8 --enable-clip \
    --batch "$B" --decode-batch "$DB" > "runs/v4/${tag}.log" 2>&1 &
  local pid=$!
  sample_vram "$vf" "$pid" &
  local sp=$!
  wait $pid
  local rc=$?
  kill $sp 2>/dev/null
  local el=$(( $(date +%s)-t0 ))
  local pk=$(sort -n "$vf" | tail -1)
  local wall=$(grep -a '\[pgt_r\] done' "runs/v4/${tag}.log" | sed 's/.*"wall_sec_render": /wall=/; s/,.*//')
  local ips=$(awk -v w="${wall#wall=}" 'BEGIN{ if (w+0>0) printf "%.2f", 750/(w+0); else print "nan" }')
  local fb=$(grep -ac 'no engine' "runs/v4/${tag}.log")
  echo "batch=$B decode=$DB tag=$tag rc=$rc total=${el}s ${wall} imgs_per_s=$ips peak_vram=${pk}mib trt_miss_lines=$fb" | tee -a "$OUT"
}

run_one 32 8   bench_b32
run_one 64 8   bench_b64
run_one 96 8   bench_b96
run_one 128 8  bench_b128
run_one 32 32  bench_b32_d32
echo "BENCH DONE" | tee -a "$OUT"
