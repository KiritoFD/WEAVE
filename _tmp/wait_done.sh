#!/usr/bin/env bash
# Block until the pgt pipeline finishes (ALL DONE / FAILED) or dies; heartbeat every 15 min.
set -u
PROG=/mnt/i/Latent_Style/WEAVE/runs/pgt/PROGRESS.md
for i in $(seq 1 900); do
  if tail -n 6 "$PROG" | grep -qE 'ALL DONE|FAILED'; then
    echo "== DONE_DETECTED =="
    tail -n 8 "$PROG"
    exit 0
  fi
  if ! pgrep -f 'pipeline2.sh' > /dev/null 2>&1; then
    echo "== PIPELINE_NOT_RUNNING =="
    tail -n 10 "$PROG"
    exit 3
  fi
  if [ $((i % 15)) -eq 0 ]; then
    echo "[$(date -u '+%F %T')] alive ${i}min | $(tail -n 1 "$PROG")"
  fi
  sleep 60
done
echo "== TIMEOUT_15H =="
exit 4
