#!/usr/bin/env bash
set -u
tmux kill-session -t pgt3 2> /dev/null
tmux new-session -d -s pgt3 "bash /mnt/i/Latent_Style/WEAVE/_tmp/pipeline_s2b.sh"
sleep 2
tmux ls
