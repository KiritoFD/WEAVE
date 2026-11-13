#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== p5 outputs present ==="
ls -la runs/v6_24h/p5_final/ 2>/dev/null || echo "  no p5_final dir"
ls -d runs/v6_24h/eval_heldout400_* runs/v6_24h/eval_same150_* 2>/dev/null
echo
echo "=== p5_final.md ==="
if [ -f runs/v6_24h/p5_final/p5_final.md ]; then
  cat runs/v6_24h/p5_final/p5_final.md
else
  echo "  missing"
fi
