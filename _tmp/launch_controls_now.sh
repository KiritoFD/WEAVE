#!/usr/bin/env bash
set -u
tmux kill-session -t pgtctrl 2> /dev/null
tmux new-session -d -s pgtctrl "bash /mnt/i/Latent_Style/WEAVE/_tmp/run_controls_now.sh"
sleep 2
tmux ls
