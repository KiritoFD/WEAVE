#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
if tmux has-session -t pgt 2>/dev/null; then
  echo "tmux session pgt already exists"
  exit 0
fi
tmux new-session -d -s pgt "bash _tmp/pipeline.sh"
echo "launched pgt session"
tmux ls
