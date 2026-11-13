#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=v11p3long
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  exit 0
fi
tmux new-session -d -s "$SESSION" -n p3long "bash -u sbflow/scripts/run_p3long.sh; echo EXIT=\$?"
sleep 2
echo "started tmux '$SESSION'"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
