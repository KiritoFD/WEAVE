#!/bin/bash
cd "$(cd "$(dirname "$0")/../.." && pwd)"
tmux kill-session -t v13 2>/dev/null
tmux new-session -d -s v13 'python3 -m exp.v13.run_p1 2>&1 | tee runs/v13/p1_run.log'
echo "TMUX_STARTED"
tmux ls
