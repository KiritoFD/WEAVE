#!/usr/bin/env bash
# Launch the b256 remediation pipeline in tmux (survives ssh drops).
set -u
tmux kill-session -t pgtb256 2> /dev/null
tmux new-session -d -s pgtb256 "bash /mnt/i/Latent_Style/WEAVE/_tmp/pipeline_b256.sh"
sleep 2
tmux ls
