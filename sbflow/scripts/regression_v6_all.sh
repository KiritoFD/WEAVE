#!/usr/bin/env bash
# Full regression suite: every test and audit built during this project. CPU only, so it is safe
# to run while training owns the GPU.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

FAILED=0
run() {
  local label=$1; shift
  echo
  echo "=================== $label ==================="
  if "$@" 2>&1 | tail -n 20; then
    :
  else
    echo "  >>> FAILED: $label"
    FAILED=$((FAILED+1))
  fi
}

run "1. Smoke16 (structural, 13 checks)"        python3 -m tools.v6_24h.smoke16
run "2. eval200 row schema (5 checks)"           python3 -m tools.v6_24h.test_row_schema
run "3. S6 style-key gate (7 checks)"            python3 -m tools.v6_24h.test_s6 --device cpu
run "4. clustered bootstrap math"                bash sbflow/scripts/test_v6_stats.sh
run "5. import preflight (15 modules)"           bash sbflow/scripts/preflight_imports.sh
run "6. data leakage audit"                      python3 -m tools.v6_24h.leakage_audit

echo
echo "=================== SUMMARY ==================="
if [ "$FAILED" -eq 0 ]; then
  echo "ALL SUITES PASSED (6/6)"
else
  echo "$FAILED SUITE(S) FAILED"
fi
exit "$FAILED"
