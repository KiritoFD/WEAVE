#!/usr/bin/env bash
# Launch the V6 24h serial queue in tmux (survives SSH drop).
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SESSION=${1:-v6_24h}
MAXH=${2:-24}
mkdir -p runs/v6_24h/train_logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi
tmux new-session -d -s "$SESSION" -n q \
  "python3 -u -m tools.v6_24h.queue --run --max-hours $MAXH 2>&1 | tee -a runs/v6_24h/queue.log; echo QUEUE_EXIT=\$?"
sleep 3
echo "started tmux '$SESSION' (max-hours=$MAXH)"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
