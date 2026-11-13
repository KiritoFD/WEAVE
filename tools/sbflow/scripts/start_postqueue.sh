#!/usr/bin/env bash
# Launch the post-queue driver in its own tmux session so it survives SSH drops.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=postq
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi
tmux new-session -d -s "$SESSION" -n q \
  "bash -u sbflow/scripts/postqueue_v6_24h.sh ${1:-}; echo POSTQUEUE_EXIT=\$? >> runs/v6_24h/postqueue.log"
sleep 3
echo "started tmux '$SESSION'"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
