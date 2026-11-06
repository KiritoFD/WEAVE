#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
tmux kill-session -t pgt 2>/dev/null
sleep 1
pkill -f build_train_cache.py 2>/dev/null
sleep 2
tmux new-session -d -s pgt "bash _tmp/pipeline2.sh"
tmux ls
