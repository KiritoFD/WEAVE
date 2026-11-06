#!/usr/bin/env bash
tmux kill-session -t v4b 2> /dev/null
tmux new-session -d -s v4b "bash /mnt/i/Latent_Style/WEAVE/_tmp/v4b_pipeline.sh"
sleep 3
tmux ls
