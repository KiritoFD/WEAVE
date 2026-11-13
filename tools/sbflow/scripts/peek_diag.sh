#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== tmux ==="
tmux list-panes -t v10diag -F '  #{pane_id} #{pane_pid} #{pane_current_command}' 2>/dev/null || echo "  no session"
echo "=== processes ==="
ps -eo pid,etime,args --no-headers | grep -E "sbflow\.(diagnose_p|score_cpu)" | grep -v grep | cut -c1-100
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== pane tail ==="
tmux capture-pane -p -S -80 -t v10diag 2>/dev/null | grep -vE '^\s*$' | tail -20
echo "=== json ==="
ls -lh runs/blog_sbflow/v10/diagnostics_04M.json 2>/dev/null || echo "  not written yet"
