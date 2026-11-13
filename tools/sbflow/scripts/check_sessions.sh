#!/usr/bin/env bash
set -uo pipefail
echo "=== tmux sessions ==="
tmux ls 2>&1 || true
echo "=== v11p1 pane ==="
tmux list-panes -t v11p1 -F '  #{pane_id} pid=#{pane_pid} cmd=#{pane_current_command}' 2>&1 || true
echo "=== v11 processes ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\.(v11_run|train5|gen5|score_cpu)" | grep -v grep | cut -c1-120 || echo "  none"
