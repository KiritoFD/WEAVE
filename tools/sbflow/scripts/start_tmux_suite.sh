#!/usr/bin/env bash
# Start the V10 suite inside a detached tmux session so SSH disconnects cannot kill it.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=${1:-v10}
SCRIPT=/mnt/i/Latent_Style/WEAVE/sbflow/scripts/launch_v10_suite.sh

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists:"
  tmux list-panes -t "$SESSION" -F '  pane #{pane_id} pid=#{pane_pid} #{pane_current_command}'
  exit 0
fi

chmod +x "$SCRIPT"
tmux new-session -d -s "$SESSION" -n suite "bash $SCRIPT; echo EXIT_CODE=\$?; sleep 86400"
sleep 2
echo "started tmux session '$SESSION'"
tmux list-panes -t "$SESSION" -F '  pane #{pane_id} pid=#{pane_pid} #{pane_current_command}'
echo "--- last lines ---"
tmux capture-pane -p -t "$SESSION" | tail -n 15
