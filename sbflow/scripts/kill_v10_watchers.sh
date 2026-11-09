#!/usr/bin/env bash
# Kill the abandoned V10 suite watchers; keep v11p1 (the running P1) untouched.
set -uo pipefail
for s in suite40 suite40_continue suite40_full750 suite40_trend; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s" && echo "  killed $s"
  else
    echo "  $s absent"
  fi
done
echo "=== remaining sessions ==="
tmux ls 2>&1 || true
