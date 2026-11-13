#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
OUT=runs/v6_24h/_p0/logs
echo "=== logs present ==="
ls -1 "$OUT" 2>/dev/null || echo "  none"
echo
echo "=== A0 equivalence (step-1 loss must be 1.3165) ==="
grep -a "step 1/2" "$OUT/P0_a0_equiv.log" 2>/dev/null || echo "  not yet"
grep -a "step1 diag" "$OUT/P0_a0_equiv.log" 2>/dev/null || echo "  no diag yet"
echo
echo "=== per-flag results ==="
for n in P0_no_style P0_no_canon P0_dim32 P0_inv_off P0_no_b1 P0_same_off \
         P0_core_off P0_ref_k256 P0_no_inject_64 P0_no_inject_32 P0_no_inject_16; do
  f="$OUT/$n.log"
  [ -f "$f" ] || { echo "  $n : not started"; continue; }
  if grep -qa "Traceback" "$f"; then
    echo "  $n : CRASH"; grep -aA2 "Error" "$f" | head -4 | sed 's/^/      /'
  elif grep -qa "step 2/2" "$f"; then
    echo "  $n : OK (2 steps done)"
    grep -a "params=" "$f" | head -1 | sed 's/^/      /'
  else
    echo "  $n : running/incomplete"
  fi
done
echo
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
