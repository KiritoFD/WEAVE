#!/usr/bin/env bash
set -uo pipefail
echo "=== all suite40 driver processes ==="
ps -eo pid,etime,args --no-headers | grep -E "python3 -u -m sbflow.suite40" | grep -v grep | cut -c1-150 || echo "  none"
echo "=== watcher scripts' wait conditions ==="
for f in continue_suite40_rest.sh run_full750_when_done.sh run_trend_when_done.sh; do
  echo "  -- $f --"
  grep -nE "terminal|pgrep|has-session|suite40 .*--arms|full750_results" \
    "/mnt/i/Latent_Style/WEAVE/sbflow/scripts/$f" 2>/dev/null | head -8 || true
done
