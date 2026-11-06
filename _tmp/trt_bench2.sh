#!/usr/bin/env bash
# TRT b8/b16/b32：decode 单图耗时 + 全链 imgs/s（40 张子集，batch64/t8）
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }
CK=runs/v4/S1_A_P5/step_000600.pt
[ -f "$CK" ] || CK=$(ls -1v runs/v4/S1_A_P5/epoch_*.pt | tail -1)
BR=/mnt/i/Latent_Style/pseudo_gt_24h/bench_render
mkdir -p "$BR"

python3 -u _tmp/trt_decode_bench.py > runs/v4/trt_decode_bench.txt 2>&1
log "TRT decode bench done"; cat runs/v4/trt_decode_bench.txt | tee -a "$PROG"

for db in 8 16 32; do
  python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$CK" --cell "bench_db${db}" \
    --render-root "$BR" --max-src-samples 8 --num-steps 8 --enable-clip \
    --batch 64 --decode-batch $db --save-threads 8 \
    > "runs/v4/bench_db${db}.log" 2>&1
  rc=$?
  res=$(grep -o '"num_images": [0-9]*, "wall_sec_render": [0-9.]*' "runs/v4/bench_db${db}.log" | tail -1)
  log "chain decode_batch=$db rc=$rc $res"
done
log "TRT bench done"
