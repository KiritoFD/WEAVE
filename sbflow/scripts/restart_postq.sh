#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
tmux kill-session -t postq 2>/dev/null && echo "killed postq" || echo "postq not running"
echo "=== finalist checkpoints ==="
for p in runs/v6/L7_highC_dim32/L7_highC_dim32.pt \
         runs/v6/L4_ref_full/L4_ref_full.pt \
         runs/v6/S6_style_keys/S6_style_keys.pt \
         runs/v6/L2_highS/L2_highS.pt; do
  [ -f "$p" ] && echo "  OK   $p ($(du -h "$p" | cut -f1))" || echo "  MISS $p"
done
echo "=== tmux ==="
tmux ls
