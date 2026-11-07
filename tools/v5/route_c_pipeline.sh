#!/usr/bin/env bash
# WEAVE V5 Route C serial decision-tree driver.
# One GPU task at a time. Appends one line per stage to runs/v5/PROGRESS.md.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
RC=runs/v5/route_c
LOGS=$RC/logs
mkdir -p "$RC" "$LOGS" runs/v5
PROG=runs/v5/PROGRESS.md
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }
run() {  # name, logfile, command...
  local name=$1 lg=$2; shift 2
  log "$name START"
  local t=$(date +%s)
  "$@" > "$LOGS/$lg" 2>&1
  local rc=$?
  log "$name END rc=$rc elapsed=$(($(date +%s)-t))s log=$LOGS/$lg"
  if [ $rc -ne 0 ]; then tail -n 20 "$LOGS/$lg" | tee -a "$PROG"; return $rc; fi
  return 0
}

# Stage 1: full-noise Rectified Flow prior (2500 steps, effective batch 256, ~10G VRAM)
run "C1_train_full" train_full.log \
  python3 -u tools/v5/route_c.py train --steps 2500 --microbatch 128 --accum 2 \
    --out "$RC/flow_full.pt" || { log "C1 FATAL"; exit 1; }

# Stage 2: pure-noise generation credibility
run "C2_noise_check" check.log \
  python3 -u tools/v5/route_c.py check --ckpt "$RC/flow_full.pt" \
    --out "$RC/noise_check.json" || { log "C2 FATAL"; exit 1; }

# Decision: if pure-noise prior is not credible, train low/mid-noise denoiser
python3 - "$RC/noise_check.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("NOISE_CREDIBLE", int(bool(r["generation_prior_credible"])), "margin", round(r["margin"],4))
raise SystemExit(0 if r["generation_prior_credible"] else 7)
PY
if [ $? -eq 7 ]; then
  log "C2 DECISION pure-noise prior failed -> lowmid denoising fallback"
  run "C2b_train_lowmid" train_lowmid.log \
    python3 -u tools/v5/route_c.py train --steps 1200 --microbatch 128 --accum 2 \
      --noise-mode lowmid --out "$RC/flow_lowmid.pt" || { log "C2b FATAL"; exit 1; }
  run "C2c_lowmid_check" check_lowmid.log \
    python3 -u tools/v5/route_c.py check --ckpt "$RC/flow_lowmid.pt" \
      --tau0 0.4 --out "$RC/noise_check_lowmid.json" || { log "C2c FATAL"; exit 1; }
  echo "{\"ckpt\":\"/mnt/i/Latent_Style/WEAVE/$RC/flow_lowmid.pt\",\"tau0\":0.55,\"guide\":false,\"seed\":42}" > "$RC/fallback_config.json"
  CKPT="$RC/flow_lowmid.pt"
  QUICK_OUT="$RC/quick_report_lowmid.json"
  # quick always rewrites best_config.json with its own --ckpt; use that outcome.
  CFG_FILE="$RC/best_config.json"
else
  CKPT="$RC/flow_full.pt"
  QUICK_OUT="$RC/quick_report.json"
  CFG_FILE="$RC/best_config.json"
fi

# Stage 3: 200-task sweep over tau0 x coarse-guidance + Style-ID selectivity
run "C3_quick_sweep" quick.log \
  python3 -u tools/v5/route_c.py quick --ckpt "$CKPT" --out "$QUICK_OUT" || { log "C3 FATAL"; exit 1; }

# If the sweep produced a best_config (full-noise path), use it; fallback path keeps fallback_config.
if [ ! -f "$CFG_FILE" ]; then echo "{\"ckpt\":\"/mnt/i/Latent_Style/WEAVE/$RC/flow_lowmid.pt\",\"tau0\":0.55,\"guide\":false,\"seed\":42}" > "$CFG_FILE"; fi
log "C3 DECISION best_config=$(cat "$CFG_FILE")"

# Stage 4: 600 cross + 150 same, two random seeds, full frozen protocol
run "C4_final_750" final.log \
  python3 -u tools/v5/route_c.py final --ckpt "$CKPT" --config "$(cat "$CFG_FILE")" || { log "C4 FATAL"; exit 1; }

# Stage 5: write the human-readable 24h delivery report (CPU only)
run "C5_summary" summary.log \
  python3 -u tools/v5/summarize_v5.py || { log "C5 FATAL"; exit 1; }

log "V5 ROUTE C ALL DONE"
