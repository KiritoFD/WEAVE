#!/usr/bin/env bash
# Launch the FULL 40-arm suite, core-first, under tmux with a 12h hard budget.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
SESSION=${1:-suite40}
ROOT=runs/blog_sbflow/suite40
mkdir -p "$ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists; leaving it untouched:"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi

tmux new-session -d -s "$SESSION" -n suite \
  "python3 -u -m sbflow.suite40 --root $ROOT --arms core --resume-suite --max-hours 4.5 2>&1 | tee -a $ROOT/logs/suite_core.log; python3 -u -m sbflow.suite40 --root $ROOT --arms rest --resume-suite --max-hours 7.5 2>&1 | tee -a $ROOT/logs/suite_rest.log; echo EXIT=\$?; sleep 86400"
sleep 3
echo "started tmux '$SESSION' arms=all (core first), 12h budget"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
