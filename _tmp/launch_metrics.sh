#!/usr/bin/env bash
# Launch the metrics rerun inside a tmux session so an ssh drop cannot kill it.
set -u
tmux kill-session -t pgtmet 2> /dev/null
tmux new-session -d -s pgtmet "bash /mnt/i/Latent_Style/WEAVE/_tmp/finish_metrics.sh"
sleep 2
tmux ls
