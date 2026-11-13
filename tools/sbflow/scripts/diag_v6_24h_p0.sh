#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
OUT=runs/v6_24h/_p0/logs
echo "=== B1 branch status across arms ==="
for n in P0_a0_equiv P0_no_b1 P0_same_off; do
  echo "  -- $n"
  grep -a "B1 branch" "$OUT/$n.log" 2>/dev/null | sed 's/^/     /'
  grep -a "step1 diag" "$OUT/$n.log" 2>/dev/null | sed 's/^/     /'
done
echo
echo "=== reference pool ==="
grep -a "reference pool" "$OUT"/P0_a0_equiv.log "$OUT"/P0_ref_k256.log 2>/dev/null | sed 's/^/  /'
echo
echo "=== lam values actually used ==="
grep -a "lam_same=" "$OUT"/P0_a0_equiv.log "$OUT"/P0_inv_off.log "$OUT"/P0_same_off.log 2>/dev/null \
  | sed 's/.*\(lam_same=[^ ]* lam_s=[^ ]* lam_c=[^ ]*\).*/  \1/'
echo
echo "=== all logs finished? ==="
for n in P0_a0_equiv P0_no_style P0_no_canon P0_dim32 P0_inv_off P0_no_b1 P0_same_off \
         P0_core_off P0_ref_k256 P0_no_inject_64 P0_no_inject_32 P0_no_inject_16; do
  f="$OUT/$n.log"
  if [ ! -f "$f" ]; then echo "  $n MISSING"; continue; fi
  if grep -qa "Traceback" "$f"; then echo "  $n CRASH"
  elif grep -qa "step 2/2" "$f"; then echo "  $n ok"
  else echo "  $n partial"; fi
done
