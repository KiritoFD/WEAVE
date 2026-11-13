#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
for n in A0_ref A1_no_content A2_no_style A3_canon_none A4_dim24 A5_dim32 A6_dim40 A7_steps400; do
  f="runs/v6/logs/${n}.log"
  [ -f "$f" ] || continue
  echo "=== $n ==="
  tail -n 4 "$f"
done
