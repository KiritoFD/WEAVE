#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== checkpoints the P5 step needs (after the finalist update) ==="
for p in runs/v6/L10_k1024_dim24/L10_k1024_dim24.pt \
         runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt \
         runs/v6/L9_k1024_dim32_lc075/L9_k1024_dim32_lc075.pt \
         runs/v6/L2_highS/L2_highS.pt \
         runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt; do
  if [ -f "$p" ]; then
    echo "  OK   $p ($(du -h "$p" | cut -f1))"
  else
    echo "  MISS $p"
  fi
done
