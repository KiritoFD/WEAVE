#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
PROG=runs/gt/PROGRESS.md
log() { echo "[$(date -u '+%F %T')] $*" | tee -a "$PROG"; }
mkdir -p runs/v4/S1_A_P5 runs/gt
free -g | head -2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
( while true; do
    echo "$(date -u +%T) vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) rss_used_mib=$(free -m | awk '/Mem:/{print $3}')"
    sleep 5; done ) > runs/v4/S1_A_P5/res.log &
echo $! > runs/v4/S1_A_P5/sampler.pid
T=$(date +%s)
python3 -u run.py --config configs/v4/S1_A_P5.json > runs/v4/S1_A_P5/train.log 2>&1
rc=$?
kill $(cat runs/v4/S1_A_P5/sampler.pid) 2>/dev/null
log "S1 train rc=$rc elapsed=$(($(date +%s)-T))s"
tail -n 25 runs/v4/S1_A_P5/train.log | cut -c1-300
pgrep -af python3 || true
