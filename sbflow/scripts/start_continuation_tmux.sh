#!/usr/bin/env bash
# Start the continuation watcher inside its own tmux session (survives SSH disconnect).
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=suite40_continue
SCRIPT=/mnt/i/Latent_Style/WEAVE/sbflow/scripts/continue_suite40_rest.sh
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi
tmux new-session -d -s "$SESSION" -n watch "bash $SCRIPT; echo EXIT=\$?; sleep 86400"
sleep 2
echo "started tmux '$SESSION'"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
