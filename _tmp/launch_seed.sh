#!/usr/bin/env bash
set -u
tmux kill-session -t pgtseed 2> /dev/null
tmux new-session -d -s pgtseed "bash /mnt/i/Latent_Style/WEAVE/_tmp/run_seed_after.sh"
sleep 2
tmux ls
