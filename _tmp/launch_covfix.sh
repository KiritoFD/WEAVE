#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
tmux kill-session -t covfix 2>/dev/null
tmux new -s covfix -d "bash /mnt/i/Latent_Style/WEAVE/_tmp/stage_covfix.sh > /mnt/i/Latent_Style/WEAVE/runs/p2/stage_covfix.log 2>&1"
sleep 3
tmux ls
tail -n 5 runs/p2/stage_covfix.log 2>/dev/null
