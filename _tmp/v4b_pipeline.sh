#!/usr/bin/env bash
# V4 续跑：S3_E3 -> S4a_E1（缓存 -> 训练 128x2/上限2000/早停 -> 渲染 b64/t8/TRT b8 -> 750 评估）
# 单 GPU 串行，不插队。每臂完成即写 arm_report.json。
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
RR=/mnt/i/Latent_Style/pseudo_gt_24h/eval_render
PAIRS=/mnt/i/Latent_Style/pseudo_gt_24h/pairs
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }

build_cache() {  # $1=spec $2=dir
  local spec=$1 dir=$2
  local t=$(date +%s)
  log "cache $spec start -> $dir"
  python3 -u tools/pseudo_gt/build_train_cache.py --arm reverse --spec "$spec" --versions 2 \
    --out "$PAIRS/$dir" > "runs/v4/build_${dir}.log" 2>&1
  local rc=$?
  log "cache $spec rc=$rc elapsed=$(($(date +%s)-t))s counts=$(python3 -c "import json;m=json.load(open('$PAIRS/$dir/manifest.json'));print(m['count'],m['shards'])" 2>/dev/null)"
  [ $rc -ne 0 ] && { tail -n 15 "runs/v4/build_${dir}.log" | cut -c1-300; return 1; }
  return 0
}

run_arm() {  # $1=name $2=cfg
  local name=$1 cfg=$2
  mkdir -p "runs/v4/$name"
  ( while true; do
      echo "$(date -u +%T) vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) rss_used_mib=$(free -m | awk '/Mem:/{print $3}')"
      sleep 5; done ) > "runs/v4/$name/res.log" &
  local sp=$!
  local t=$(date +%s)
  python3 -u run.py --config "$cfg" > "runs/v4/$name/train.log" 2>&1
  local rc=$?
  kill $sp 2>/dev/null
  local el=$(( $(date +%s)-t ))
  local pk=$(awk -F'vram_mib=' '{split($2,a," "); if(a[1]+0>vm)vm=a[1]+0} END{print vm}' "runs/v4/$name/res.log")
  local rk=$(awk -F'rss_used_mib=' '{v=$2+0; if(v>m)m=v} END{print m}' "runs/v4/$name/res.log")
  local stop=$(grep -E 'Reached stop_after_global_steps|Internal early stop summary' "runs/v4/$name/train.log" | tail -n 2 | tr '\n' ';' | cut -c1-300)
  log "$name train rc=$rc elapsed=${el}s peak_vram=${pk}mib peak_rss=${rk}mib | $stop"
  [ $rc -ne 0 ] && { tail -n 12 "runs/v4/$name/train.log" | cut -c1-300; return 1; }
  echo "$el $pk $rk" > "runs/v4/$name/train_facts.txt"
  return 0
}

eval_arm() {  # $1=name
  local name=$1
  local ck
  ck=$(ls -t runs/v4/$name/*.pt 2>/dev/null | head -n 1)
  [ -z "$ck" ] && { log "$name NO CKPT"; return 1; }
  local cell="${name}_final"
  if [ ! -f "$RR/$cell/metrics.csv" ] || [ "$(tail -n +2 "$RR/$cell/metrics.csv" | wc -l)" -lt 750 ]; then
    local t=$(date +%s)
    python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$ck" --cell "$cell" \
      --render-root "$RR" --max-src-samples 30 --num-steps 8 --enable-clip \
      --batch 64 --decode-batch 8 --save-threads 8 \
      > "runs/v4/$name/render.log" 2>&1
    if [ $? -ne 0 ]; then log "$name render FAILED"; tail -n 12 "runs/v4/$name/render.log" | cut -c1-300; return 1; fi
    log "$name render done elapsed=$(($(date +%s)-t))s"
  fi
  python3 -u tools/pseudo_gt/v4_eval750.py --cell "$cell" --render-root "$RR" --max-src 30 \
    > "runs/v4/$name/eval750.log" 2>&1
  local rc=$?
  log "$name eval750 ckpt=$(basename $ck) rc=$rc"
  [ $rc -ne 0 ] && { tail -n 15 "runs/v4/$name/eval750.log" | cut -c1-300; return 1; }
  python3 -u _tmp/v4b_report.py "$name" > "runs/v4/$name/arm_report.txt" 2>&1
  log "$name REPORT"; cat "runs/v4/$name/arm_report.txt" | tee -a "$PROG"
  return 0
}

log "V4B pipeline start (S3_E3 -> S4a_E1), GPU serial"
mkdir -p "$PAIRS/_partial"
for d in reverse_e3 reverse_e1; do
  if [ -d "$PAIRS/$d" ]; then mv "$PAIRS/$d" "$PAIRS/_partial/${d}_smoke_partial" 2>/dev/null; fi
done

if build_cache E3 reverse_e3; then
  run_arm S3_E3 configs/v4/S3_E3.json && eval_arm S3_E3 || log "S3 FAILED"
else log "S3 cache FAILED-FATAL"; fi

if build_cache E1 reverse_e1; then
  run_arm S4a_E1 configs/v4/S4a_E1.json && eval_arm S4a_E1 || log "S4a FAILED"
else log "S4a cache FAILED-FATAL"; fi

log "V4B arms done; TRT b16/b32 bench queued"
bash _tmp/trt_bench2.sh
log "V4B ALL DONE"
