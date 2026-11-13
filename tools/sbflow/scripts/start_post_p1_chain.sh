#!/usr/bin/env bash
# Launch the post-P1 chain in its own tmux session.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=v11chain
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi
tmux new-session -d -s "$SESSION" -n chain "bash -u sbflow/scripts/post_p1_chain.sh; echo EXIT=\$?"
sleep 2
echo "started tmux '$SESSION'"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
