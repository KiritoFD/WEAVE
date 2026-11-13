#!/usr/bin/env bash
# Start (or restart) the P0-P3 diagnostics inside a detached tmux session.
set -uo pipefail
SESSION=${1:-v10diag}
SCRIPT=/mnt/i/Latent_Style/WEAVE/sbflow/scripts/run_diagnostics_p.sh
LOG=/mnt/i/Latent_Style/WEAVE/runs/blog_sbflow/v10/logs/diag.log
mkdir -p "$(dirname "$LOG")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  sleep 2
fi
tmux new-session -d -s "$SESSION" -n diag \
  "bash $SCRIPT p0,p1,p2,p3 2>&1 | tee -a $LOG; echo EXIT=\$?; sleep 86400"
sleep 3
tmux list-panes -t "$SESSION" -F '  pane #{pane_id} pid=#{pane_pid} #{pane_current_command}'
