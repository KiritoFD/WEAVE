#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }
RR=/mnt/i/Latent_Style/pseudo_gt_24h/eval_render

run_arm() {  # $1=name $2=config
  local name=$1 cfg=$2
  mkdir -p "runs/v4/$name"
  rm -f "runs/v4/$name"/epoch_*.pt "runs/v4/$name"/step_*.pt "runs/v4/$name"/best.pt 2>/dev/null
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
  local stop=$(grep -E 'Internal early stop summary|stop_after_global_steps' "runs/v4/$name/train.log" | tail -n 2 | tr '\n' ';' | cut -c1-300)
  log "$name train rc=$rc elapsed=${el}s peak_vram=${pk}mib peak_rss=${rk}mib | $stop"
  if [ $rc -ne 0 ]; then tail -n 12 "runs/v4/$name/train.log" | cut -c1-300; return 1; fi
}

eval_arm() {  # $1=name ; renders only if PNGs missing
  local name=$1
  local ck
  ck=$(ls -t runs/v4/$name/*.pt 2>/dev/null | head -n 1)
  [ -z "$ck" ] && { log "$name NO CKPT"; return 1; }
  local cell="${name}_final"
  local t=$(date +%s)
  if [ ! -f "$RR/$cell/metrics.csv" ] || [ "$(tail -n +2 "$RR/$cell/metrics.csv" | wc -l)" -lt 750 ]; then
    python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$ck" --cell "$cell" \
      --render-root "$RR" --max-src-samples 30 --num-steps 8 --enable-clip \
      > "runs/v4/$name/render.log" 2>&1
    if [ $? -ne 0 ]; then log "$name render FAILED"; tail -n 12 "runs/v4/$name/render.log" | cut -c1-300; return 1; fi
  fi
  local tr=$(($(date +%s)-t))
  python3 -u tools/pseudo_gt/v4_eval750.py --cell "$cell" --render-root "$RR" --max-src 30 \
    > "runs/v4/$name/eval750.log" 2>&1
  local rc=$?
  log "$name eval750 ckpt=$(basename $ck) render=${tr}s eval_rc=$rc"
  if [ $rc -ne 0 ]; then tail -n 15 "runs/v4/$name/eval750.log" | cut -c1-300; return 1; fi
}

log "V4 chain2 start (S1 eval->S2 train->S2 eval)"
eval_arm S1_A_P5
run_arm S2_E7 configs/v4/S2_E7.json || log "S2 FAILED-FATAL"
eval_arm S2_E7
log "V4 chain2 done"
