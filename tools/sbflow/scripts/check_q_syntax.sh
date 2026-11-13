#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
FAIL=0
for f in sbflow/scripts/q_lib.sh sbflow/scripts/q1_sweep.sh \
         sbflow/scripts/q2_mechanisms.sh sbflow/scripts/q3_final.sh \
         sbflow/scripts/run_v6_next_q.sh sbflow/scripts/run_v6_next_p2.sh; do
  if bash -n "$f" 2>/dev/null; then
    echo "  OK        $f"
  else
    echo "  SYNTAX    $f"
    bash -n "$f" 2>&1 | head -3
    FAIL=$((FAIL+1))
  fi
done
echo
[ "$FAIL" -eq 0 ] && echo "ALL SCRIPTS PARSE" || echo "$FAIL script(s) failed to parse"
exit "$FAIL"
