#!/usr/bin/env bash
# Launch the 12h / 40-arm suite in a detached tmux session (survives SSH disconnect).
# Default: the nine core controls named in the execution book.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=${1:-suite40}
ARMS=${2:-core}
ROOT=runs/blog_sbflow/suite40
mkdir -p "$ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists:"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi

tmux new-session -d -s "$SESSION" -n suite \
  "python3 -u -m sbflow.suite40 --root $ROOT --arms $ARMS --resume-suite 2>&1 | tee -a $ROOT/logs/suite.log; echo EXIT=\$?"
sleep 3
echo "started tmux '$SESSION' arms=$ARMS"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
