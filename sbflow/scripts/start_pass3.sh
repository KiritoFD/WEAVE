#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=pass3
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi
tmux new-session -d -s "$SESSION" -n q \
  "bash -u sbflow/scripts/pass3_v6_24h.sh; echo PASS3_EXIT=\$? >> runs/v6_24h/pass3.log"
sleep 3
echo "started tmux '$SESSION'"
tmux ls
