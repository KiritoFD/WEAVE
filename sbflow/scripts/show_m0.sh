#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
for d in runs/v6_24h/m0_main runs/v6_24h/m0_extrap; do
  echo "================= $d ================="
  if [ -f "$d/m0_alpha.md" ]; then
    cat "$d/m0_alpha.md"
  else
    echo "  missing m0_alpha.md"
  fi
  echo
done
