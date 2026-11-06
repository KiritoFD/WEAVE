#!/usr/bin/env bash
# 通用 tmux 启动器：launch_gen.sh <session> <script>
set -u
tmux kill-session -t "$1" 2>/dev/null || true
tmux new-session -d -s "$1" "bash $2; sleep 5"
echo "STARTED $1"
