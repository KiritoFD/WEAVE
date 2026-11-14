#!/bin/bash
# V13 tmux launcher (runs inside WSL)
cd /mnt/i/Latent_Style/WEAVE
tmux kill-server 2>/dev/null
tmux new-session -d -s v13 \
  'python3 -m exp.v13.run_p1 2>&1 | tee /mnt/i/Latent_Style/WEAVE/runs/v13/p1_run.log'
sleep 5
echo "TMUX:"
tmux ls
echo "PANE:"
tmux capture-pane -t v13 -p | tail -5
