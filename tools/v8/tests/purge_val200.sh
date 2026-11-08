#!/usr/bin/env bash
# Purge all invalid val200 artefacts (that pool overlapped the latent training cache).
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE/runs/v8
shopt -s nullglob
files=(*val200*)
if [ ${#files[@]} -eq 0 ]; then
  echo "no val200 artefacts in runs/v8"
else
  for f in "${files[@]}"; do echo "REMOVING $f"; rm -f -- "$f"; done
fi
rm -f /mnt/i/Latent_Style/WEAVE/docs/v8/V8_RESULTS_val200.md
echo "done"
