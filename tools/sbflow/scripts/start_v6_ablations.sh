#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=v6abl
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi
tmux new-session -d -s "$SESSION" -n abl "bash -u sbflow/scripts/run_v6_ablations.sh; echo EXIT=\$?"
sleep 3
echo "started tmux '$SESSION'"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
