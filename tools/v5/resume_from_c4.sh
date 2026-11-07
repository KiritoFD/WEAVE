#!/usr/bin/env bash
# Frozen recovery driver: resume only the failed C4/C5 stages.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
RC=runs/v5/route_c
LOGS=$RC/logs
PROG=runs/v5/PROGRESS.md
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }
run() { local n=$1 l=$2; shift 2; log "RESUME $n START"; local t=$(date +%s); "$@" > "$LOGS/$l" 2>&1; local r=$?; log "RESUME $n END rc=$r elapsed=$(($(date +%s)-t))s log=$LOGS/$l"; if [ $r -ne 0 ]; then tail -n 25 "$LOGS/$l" | tee -a "$PROG"; fi; return $r; }

CKPT="$RC/flow_lowmid.pt"
CFG=$(cat "$RC/best_config.json")
log "RESUME using ckpt=$CKPT config=$CFG"

run "C4_final_750" final_resume.log \
  python3 -u tools/v5/route_c.py final --ckpt "$CKPT" --config "$CFG" || { log "RESUME C4 FATAL"; exit 1; }

run "C5_summary" summary_resume.log \
  python3 -u tools/v5/summarize_v5.py || { log "RESUME C5 FATAL"; exit 1; }

log "V5 ROUTE C ALL DONE (RESUMED)"
