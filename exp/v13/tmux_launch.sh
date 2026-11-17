#!/bin/bash
# V13 tmux launcher (runs inside WSL)
cd "$(cd "$(dirname "$0")/../.." && pwd)"
tmux kill-session -t v13 2>/dev/null || true
tmux new-session -d -s v13 \
  'python3 -m exp.v13.run_p1 2>&1 | tee runs/v13/p1_run.log'
sleep 5
echo "TMUX:"
tmux ls
echo "PANE:"
tmux capture-pane -t v13 -p | tail -5
