#!/usr/bin/env bash
# Assert the execution book's single-GPU-process constraint and report any violation.
set -uo pipefail
echo "=== GPU processes (train5 / gen5) ==="
ps -eo pid,etime,args --no-headers | grep -E "python3 -m sbflow\.(train5|gen5)|/usr/bin/python3 -m sbflow\.(train5|gen5)" \
  | grep -v grep | cut -c1-130 || true
N=$(ps -eo args --no-headers | grep -E "python3 -m sbflow\.(train5|gen5)|/usr/bin/python3 -m sbflow\.(train5|gen5)" \
  | grep -v grep | wc -l)
echo "  count=$N"
if [ "$N" -gt 1 ]; then
  echo "  VIOLATION: more than one GPU train/generate process"
  exit 1
fi
echo "=== CPU scorers (allowed to overlap) ==="
M=$(ps -eo args --no-headers | grep -E "python3 -m sbflow.score_cpu|/usr/bin/python3 -m sbflow.score_cpu" \
  | grep -v grep | wc -l)
echo "  score_cpu count=$M"
echo "SINGLE-GPU CONSTRAINT OK"
