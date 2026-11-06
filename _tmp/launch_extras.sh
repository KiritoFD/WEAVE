#!/usr/bin/env bash
set -u
tmux kill-session -t pgtextra 2> /dev/null
tmux new-session -d -s pgtextra "bash /mnt/i/Latent_Style/WEAVE/_tmp/run_extras_after.sh"
sleep 2
tmux ls
