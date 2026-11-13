#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== tmux ==="
tmux ls
echo "=== postqueue log ==="
if [ -f runs/v6_24h/postqueue.log ]; then
  tail -n 6 runs/v6_24h/postqueue.log
else
  echo "  (no log yet)"
fi
echo "=== queue still running? ==="
pgrep -f "tools.v6_24h.queue" >/dev/null && echo "  yes" || echo "  no"
