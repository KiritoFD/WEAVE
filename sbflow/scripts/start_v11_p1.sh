#!/usr/bin/env bash
# V11 P1: four corrected arms, strictly serial, detached in tmux.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SESSION=${1:-v11p1}
ROOT=runs/blog_sbflow/v11
mkdir -p "$ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi

tmux new-session -d -s "$SESSION" -n p1 \
  "python3 -u -m sbflow.v11_run --arms d1,d2,e1,e2 --threads 4 2>&1 | tee -a $ROOT/logs/p1.log; echo EXIT=\$?"
sleep 3
echo "started tmux '$SESSION'"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
