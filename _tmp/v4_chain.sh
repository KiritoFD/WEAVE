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
  local stop=$(grep -E 'early stop|EarlyStop|早停|stop requested|stop_after' "runs/v4/$name/train.log" | tail -n 2 | tr '\n' ';' | cut -c1-360)
  log "$name train rc=$rc elapsed=${el}s peak_vram=${pk}mib peak_rss=${rk}mib | $stop"
  if [ $rc -ne 0 ]; then tail -n 12 "runs/v4/$name/train.log" | cut -c1-300; return 1; fi
  echo "$name" > /tmp/v4_arm_name
}

eval_arm() {  # $1=name
  local name=$1
  local ck
  ck=$(ls -t runs/v4/$name/*.pt 2>/dev/null | head -n 1)
  if [ -z "$ck" ]; then log "$name NO CKPT"; return 1; fi
  local cell="${name}_final"
  local t=$(date +%s)
  python3 -u tools/pseudo_gt/pgt_render.py --checkpoint "$ck" --cell "$cell" \
    --render-root "$RR" --max-src-samples 30 --num-steps 8 --enable-clip \
    > "runs/v4/$name/render.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "$name render FAILED rc=$rc"; tail -n 12 "runs/v4/$name/render.log" | cut -c1-300; return 1; fi
  local t2=$(date +%s)
  python3 -u tools/pseudo_gt/v4_eval750.py --cell "$cell" --render-root "$RR" --max-src 30 \
    > "runs/v4/$name/eval750.log" 2>&1
  rc=$?
  log "$name ckpt=$(basename $ck) render=$(($(date +%s)-t))s eval_rc=$rc"
  if [ $rc -ne 0 ]; then tail -n 15 "runs/v4/$name/eval750.log" | cut -c1-300; return 1; fi
  cat "runs/v4/$name/eval750.log" | tail -n 80
}

log "V4 chain start"
run_arm S1_A_P5 configs/v4/S1_A_P5.json || log "S1 FAILED-FATAL"
eval_arm S1_A_P5
run_arm S2_E7 configs/v4/S2_E7.json || log "S2 FAILED-FATAL"
eval_arm S2_E7
log "V4 chain: S1/S2 done; E3/E1 caches queued separately"
