#!/usr/bin/env bash
# Run the V11 P0 verification battery on the 3060.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
FAIL=0
for t in test_swd_grad test_score_aggregate test_v11_math_fixes \
         test_v11_dist_directions test_v11_online_directions \
         test_v10_math test_v10_de test_suite40_compat; do
  echo "########## $t ##########"
  if python3 -u "sbflow/tests/$t.py" 2>&1 | tail -4; then :; else FAIL=$((FAIL+1)); fi
done
echo "########## V11 P0 BATTERY DONE (failures=$FAIL) ##########"
